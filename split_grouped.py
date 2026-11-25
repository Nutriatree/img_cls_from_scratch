"""Create a conservative perceptual-similarity split for manual review."""

import csv
import html
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'raw'
OUTPUT = ROOT / 'data_grouped'
SEED = 42
HASH_DISTANCE = 18


def perceptual_hash(path):
    with Image.open(path) as image:
        pixels = np.asarray(ImageOps.exif_transpose(image).convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    indices = np.arange(32)
    basis = np.cos(np.pi * (2 * indices[None, :] + 1) * np.arange(8)[:, None] / 64)
    basis[0] /= np.sqrt(2)
    values = (basis @ pixels @ basis.T).ravel()[1:]
    return sum(int(value > np.median(values)) << i for i, value in enumerate(values))


def build_groups(paths):
    hashes = [perceptual_hash(path) for path in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = []
    for i, first in enumerate(paths):
        for j in range(i + 1, len(paths)):
            second = paths[j]
            if first.stem.split('_')[0] != second.stem.split('_')[0]:
                continue
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= HASH_DISTANCE:
                candidates.append((first.name, second.name, distance))
                parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i, path in enumerate(paths):
        groups[find(i)].append(path)
    return list(groups.values()), candidates


def main():
    paths = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError(f'No images found in {SOURCE}')
    if OUTPUT.exists():
        raise FileExistsError(f'Output already exists: {OUTPUT}')
    groups, candidates = build_groups(paths)
    classes = defaultdict(list)
    for group in groups:
        classes[group[0].stem.split('_')[0]].append(group)
    rng = random.Random(SEED)
    rows = []
    counts = {}
    for label, label_groups in sorted(classes.items()):
        rng.shuffle(label_groups)
        label_groups.sort(key=len, reverse=True)
        total = sum(map(len, label_groups))
        targets = {'train': total * .8, 'validation': total * .1, 'test': total * .1}
        assigned = Counter()
        for group in label_groups:
            split = max(targets, key=lambda name: (targets[name] - assigned[name]) / targets[name])
            assigned[split] += len(group)
            group_id = min(path.stem for path in group)
            for path in group:
                rows.append((split, label, path.name, group_id))
        counts[label] = dict(assigned)

    OUTPUT.mkdir()
    for split, label, name, group_id in rows:
        destination = OUTPUT / split / label
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, destination / name)
    with (OUTPUT / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('split', 'class', 'filename', 'group_id'))
        writer.writerows(sorted(rows))
    with (OUTPUT / 'similar_pairs.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('image_1', 'image_2', 'phash_distance'))
        writer.writerows(candidates)
    with (OUTPUT / 'review.html').open('w', encoding='utf-8') as file:
        file.write('<!doctype html><meta charset="utf-8"><title>Similar image candidates</title>')
        file.write('<h1>Similar image candidates</h1><p>Review visually; same-session images can be missed.</p>')
        for first, second, distance in sorted(candidates, key=lambda row: (row[2], row[0], row[1])):
            file.write(f'<div><p>{html.escape(first)} / {html.escape(second)} — pHash distance {distance}</p>')
            for name in (first, second):
                file.write(f'<img src="../raw/{html.escape(name, quote=True)}" width="220" alt="{html.escape(name, quote=True)}"> ')
            file.write('</div>')
    group_sizes = Counter(map(len, groups))
    print('groups by size:', dict(sorted(group_sizes.items())))
    print('candidate pairs:', len(candidates))
    print('counts:', counts)
    print('totals:', dict(Counter(row[0] for row in rows)))


if __name__ == '__main__':
    main()
