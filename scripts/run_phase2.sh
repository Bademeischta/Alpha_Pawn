#!/usr/bin/env bash
python -m src.training.train_selfplay \
  --config config/selfplay.yaml \
  --log_dir logs/phase2 \
  --ckpt_dir checkpoints/phase2 \
  --small_ckpt checkpoints/phase1/epoch_10.pt
