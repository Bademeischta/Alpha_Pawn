#!/usr/bin/env bash
python -m src.training.train_supervised \
  --config config/default.yaml \
  --log_dir logs/phase1 \
  --ckpt_dir checkpoints/phase1 \
  --data data/train_data.npz
