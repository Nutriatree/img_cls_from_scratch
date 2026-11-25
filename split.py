"""Split building images into reproducible, class-stratified datasets."""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def allocate(counts, total):
    """Allocate an integer total proportionally using largest remainders."""
    population = sum(counts.values())
    quotas = {label: total * count / population for label, count in counts.items()}
    result = {label: int(quota) for label, quota in quotas.items()}
    for label in sorted(counts, key=lambda label: (-(quotas[label] - result[label]), label))[:total - sum(result.values())]:
        result[label] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    images = sorted(path for path in args.source.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not images:
        parser.error(f'No images found in {args.source}')
    groups = defaultdict(list)
    for image in images:
        groups[image.stem.split('_')[0]].append(image)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Output directory is not empty: {args.output}')

    counts = {label: len(paths) for label, paths in groups.items()}
    train_total = int(len(images) * 0.8)
    val_total = int(len(images) * 0.1)
    train_counts = allocate(counts, train_total)
    remaining = {label: counts[label] - train_counts[label] for label in counts}
    val_counts = allocate(remaining, val_total)
    rng = random.Random(args.seed)
    rows = []
    for label in sorted(groups):
        paths = groups[label][:]
        rng.shuffle(paths)
        train_end = train_counts[label]
        val_end = train_end + val_counts[label]
        for split, subset in (('train', paths[:train_end]), ('validation', paths[train_end:val_end]), ('test', paths[val_end:])):
            folder = args.output / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for image in subset:
                shutil.copy2(image, folder / image.name)
                rows.append((split, label, image.name))
    with (args.output / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('split', 'class', 'filename'))
        writer.writerows(rows)
    for label in sorted(groups):
        print(label, dict(Counter(split for split, cls, _ in rows if cls == label)))
    print('Total:', dict(Counter(split for split, _, _ in rows)))


if __name__ == '__main__':
    main()
