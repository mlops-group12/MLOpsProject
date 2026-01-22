#!/usr/bin/env bash
set -euo pipefail

dvc pull data/train_data
# dvc pull data/celeba   # uncomment if you need it too

exec uv run python -m src.mlopsproject.train wandb.enabled=false
