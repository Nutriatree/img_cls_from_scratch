#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
eval "$(conda shell.bash hook)"
conda activate AutoAI
exec "$CONDA_PREFIX/bin/python" -m torch.distributed.run --standalone --nproc-per-node=2 load.py
