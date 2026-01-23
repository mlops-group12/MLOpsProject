#!/usr/bin/env bash
set -euo pipefail

cd /app
export DVC_NO_SCM=1

dvc --no-scm pull data/train_data
# dvc pull data/celeba   # uncomment if you need it too

exec uv run python -m src.mlopsproject.train wandb.enabled=false
