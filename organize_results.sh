#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

destination="results/legacy"
files=(model.pt model.config.json model.history.csv model.loss.png
       model_60.pt model_60.config.json model_60.history.csv model_60.loss.png
       model_grouped_60.pt model_grouped_60.config.json model_grouped_60.history.csv model_grouped_60.loss.png)

for file in "${files[@]}"; do
    if [[ -e "$file" && -e "$destination/$file" ]]; then
        printf 'Destination already exists: %s\n' "$destination/$file" >&2
        exit 1
    fi
done
mkdir -p -- "$destination"
for file in "${files[@]}"; do
    if [[ -e "$file" ]]; then
        mv -- "$file" "$destination/$file"
        printf 'Moved %s -> %s/\n' "$file" "$destination"
    fi
done
