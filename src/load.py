"""Train DenseNet-BC on the pre-split building dataset."""

import csv
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from model import DenseNetBC
from preprocessing import AugmentationConfig, Preprocessor, seed_worker, training_statistics

# Training settings
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data_grouped'
MODEL_PATH = ROOT / 'results' / 'grouped_60' / 'model.pt'
IMAGE_SIZE = (224, 224)
SEED = 42
EPOCHS = 60
WARMUP_EPOCHS = 2
USE_DDP = True  # True: launch with torchrun; BATCH_SIZE is global
BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
DEVICE = 'auto'  # auto, cpu, cuda, or mps


class ImageDataset(Dataset):
    def __init__(self, directory, classes, transform):
        self.transform = transform
        self.samples = []
        folders = sorted(path.name for path in directory.iterdir() if path.is_dir())
        if folders != classes:
            raise ValueError(f'Class folders in {directory} do not match training classes')
        for label, name in enumerate(classes):
            images = sorted(path for path in (directory / name).iterdir()
                            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            if not images:
                raise ValueError(f'No images found in {directory / name}')
            self.samples.extend((path, label) for path in images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), label


def run_epoch(model, loader, criterion, device, optimizer=None, description="Epoch", distributed=False, show_progress=True):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    confusion = torch.zeros((5, 5), dtype=torch.int64)
    progress = tqdm(loader, desc=description, unit="batch", leave=False, disable=not show_progress)
    with torch.set_grad_enabled(training):
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; training stopped')
            if training:
                loss.backward()
                optimizer.step()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            indices = (targets * 5 + predictions).cpu()
            confusion += torch.bincount(indices, minlength=25).reshape(5, 5)
            progress.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.3f}")
    if distributed:
        totals = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        total_loss, correct, total = totals.tolist()
        confusion = confusion.to(device)
        dist.all_reduce(confusion)
        confusion = confusion.cpu()
    return total_loss / total, correct / total, confusion.tolist()


def save_loss_curve(history_path, curve_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with history_path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row['epoch']) for row in rows]
    fig, ax = plt.subplots()
    ax.plot(epochs, [float(row['train_loss']) for row in rows], label='Train (label smoothing)')
    ax.plot(epochs, [float(row['validation_loss']) for row in rows], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training and validation loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(dir=curve_path.parent, suffix='.png', delete=False) as file:
        temporary_path = Path(file.name)
    try:
        fig.savefig(temporary_path, dpi=150)
        os.replace(temporary_path, curve_path)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)


def classification_metrics(confusion, classes):
    scores = {}
    for index, name in enumerate(classes):
        correct = confusion[index][index]
        actual = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * correct / (actual + predicted) if actual + predicted else 0.0
        scores[name] = {'correct': correct, 'total': actual, 'precision': precision,
                        'recall': recall, 'f1': f1}
    return scores, sum(score['f1'] for score in scores.values()) / len(classes)


def save_confusion_matrix(confusion, classes, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    image = ax.imshow(confusion, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel='Predicted', ylabel='Actual',
           title='Test confusion matrix')
    midpoint = max(max(row) for row in confusion) / 2
    for i, row in enumerate(confusion):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value > midpoint else 'black')
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.png', delete=False) as file:
        temporary_path = Path(file.name)
    try:
        fig.savefig(temporary_path, dpi=150)
        os.replace(temporary_path, destination)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)


def main():
    if USE_DDP:
        if 'RANK' not in os.environ or 'LOCAL_RANK' not in os.environ:
            raise RuntimeError('USE_DDP=True requires torchrun --nproc-per-node=2 src/load.py')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if world_size < 2:
            raise ValueError('DDP requires at least two processes')
        if DEVICE == 'cpu':
            device = torch.device('cpu')
            backend = 'gloo'
        else:
            if DEVICE not in ('auto', 'cuda') or not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
                raise RuntimeError('DDP requires a CUDA device for each process, or DEVICE=cpu for testing')
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            backend = 'nccl'
        dist.init_process_group(backend=backend)
    else:
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise RuntimeError('USE_DDP=False: use python src/load.py rather than torchrun')
        rank, world_size = 0, 1
        if DEVICE == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            device = torch.device(DEVICE)
    try:
        train(rank, world_size, device)
    finally:
        if USE_DDP and dist.is_initialized():
            dist.destroy_process_group()


def train(rank, world_size, device):
    config_path = MODEL_PATH.with_suffix('.config.json')
    history_path = MODEL_PATH.with_suffix('.history.csv')
    curve_path = MODEL_PATH.with_suffix('.loss.png')
    test_path = MODEL_PATH.with_suffix('.test.json')
    confusion_path = MODEL_PATH.with_suffix('.confusion.png')
    for path in (MODEL_PATH, config_path, history_path, curve_path, test_path, confusion_path):
        if path.exists():
            raise FileExistsError(f'{path} already exists; choose another MODEL_PATH to avoid overwriting it')
    if not 0 <= WARMUP_EPOCHS < EPOCHS:
        raise ValueError('WARMUP_EPOCHS must be non-negative and less than EPOCHS')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if BATCH_SIZE <= 0 or BATCH_SIZE % world_size:
        raise ValueError('BATCH_SIZE must be positive and divisible by the process count')

    train_dir, val_dir = DATA_DIR / 'train', DATA_DIR / 'validation'
    classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if len(classes) != 5:
        raise ValueError(f'Expected 5 class folders, found {len(classes)}')
    config = AugmentationConfig(size=IMAGE_SIZE)
    statistics = [training_statistics(train_dir, size=IMAGE_SIZE) if rank == 0 else None]
    if USE_DDP:
        dist.broadcast_object_list(statistics, src=0)
    mean, std = statistics[0]
    train_data = ImageDataset(train_dir, classes, Preprocessor(mean, std, config, training=True))
    val_data = ImageDataset(val_dir, classes, Preprocessor(mean, std, config, training=False))
    test_data = ImageDataset(DATA_DIR / 'test', classes, Preprocessor(mean, std, config, training=False))
    if USE_DDP and len(train_data) % world_size:
        raise ValueError('Training sample count must be divisible by the process count to avoid duplicated images')
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED) if USE_DDP else None
    local_batch = BATCH_SIZE // world_size
    train_loader = DataLoader(train_data, batch_size=local_batch, shuffle=sampler is None, sampler=sampler,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED + rank))
    val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED + 1))
    test_loader = DataLoader(test_data, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED + 2))
    model = DenseNetBC().to(device)
    if USE_DDP:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    base_model = model.module if USE_DDP else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_factor(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    val_loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')
    metadata = {
        'model': 'DenseNetBC', 'parameters': sum(p.numel() for p in model.parameters()),
        'data_dir': str(DATA_DIR.resolve()), 'train_samples': len(train_data),
        'validation_samples': len(val_data), 'test_samples': len(test_data),
        'classes': classes, 'mean': mean, 'std': std,
        'augmentation': asdict(config), 'seed': SEED, 'epochs': EPOCHS,
        'warmup_epochs': WARMUP_EPOCHS, 'batch_size': BATCH_SIZE, 'local_batch_size': local_batch,
        'validation_batch_size': VAL_BATCH_SIZE,
        'use_ddp': USE_DDP, 'world_size': world_size, 'num_workers': NUM_WORKERS,
        'validation_interval': 'epoch',
        'optimizer': 'AdamW', 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'loss': 'CrossEntropyLoss', 'label_smoothing': LABEL_SMOOTHING,
        'validation_label_smoothing': 0.0, 'scheduler': 'linear warmup + cosine',
        'device': str(device), 'torch_version': str(torch.__version__),
    }
    if rank == 0:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open('x', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        with history_path.open('x', newline='', encoding='utf-8') as file:
            csv.writer(file).writerow(['epoch', 'global_step', 'learning_rate', 'train_loss', 'train_accuracy',
                                       'validation_loss', 'validation_accuracy', 'validation_macro_f1', 'validation_confusion_matrix'])
        print(f'Device: {device}; processes: {world_size}; train: {len(train_data)}; validation: {len(val_data)}', flush=True)
        print(f'Classes: {classes}; mean: {mean}; std: {std}', flush=True)
    global_step = 0
    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc, _ = run_epoch(model, train_loader, train_loss_fn, device, optimizer,
                                            f'Epoch {epoch + 1}/{EPOCHS} train',
                                            distributed=USE_DDP, show_progress=rank == 0)
        global_step += len(train_loader)
        if rank == 0:
            # Only rank 0 evaluates the full validation set using the unwrapped model.
            val_loss, val_acc, val_confusion = run_epoch(base_model, val_loader, val_loss_fn, device,
                                                         description=f'Epoch {epoch + 1} validation')
            _, val_macro_f1 = classification_metrics(val_confusion, classes)
            with history_path.open('a', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow([epoch + 1, global_step, lr, train_loss, train_acc, val_loss, val_acc,
                                           val_macro_f1, json.dumps(val_confusion)])
            if val_loss < best_loss:
                checkpoint = {'model_state_dict': base_model.state_dict(), 'classes': classes,
                              'mean': mean, 'std': std, 'image_size': IMAGE_SIZE,
                              'epoch': epoch + 1, 'global_step': global_step, 'validation_loss': val_loss}
                with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix='.pt', delete=False) as file:
                    temporary_path = Path(file.name)
                try:
                    torch.save(checkpoint, temporary_path)
                    os.replace(temporary_path, MODEL_PATH)
                finally:
                    temporary_path.unlink(missing_ok=True)
                best_loss = val_loss
            save_loss_curve(history_path, curve_path)
            tqdm.write(f'Epoch {epoch + 1}/{EPOCHS} step={global_step} lr={lr:.7f} '
                       f'train_loss={train_loss:.4f} train_acc={train_acc:.3f} '
                       f'val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.3f}')
        if USE_DDP:
            dist.barrier()
        scheduler.step()
    if rank == 0:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, confusion = run_epoch(base_model, test_loader, val_loss_fn, device,
                                                    description='Best model test')
        class_metrics, macro_f1 = classification_metrics(confusion, classes)
        result = {'checkpoint': str(MODEL_PATH), 'best_epoch': checkpoint['epoch'],
                  'best_global_step': checkpoint['global_step'], 'validation_loss': checkpoint['validation_loss'],
                  'test_loss': test_loss, 'test_accuracy': test_acc, 'test_macro_f1': macro_f1, 'test_samples': len(test_data),
                  'classes': classes, 'class_metrics': class_metrics, 'confusion_matrix': confusion}
        save_confusion_matrix(confusion, classes, confusion_path)
        with test_path.open('x', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(f'Best validation loss: {best_loss:.4f}; saved: {MODEL_PATH}', flush=True)
        print(f'Test loss: {test_loss:.4f}; accuracy: {test_acc:.4f}; macro F1: {macro_f1:.4f}; saved: {test_path}', flush=True)
    if USE_DDP:
        dist.barrier()


if __name__ == '__main__':
    main()
