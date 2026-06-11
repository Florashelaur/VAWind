#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-zhongshan_10m.csv}
TARGET=${TARGET:-wind_speed_10m}
GPU=${GPU:-0}
USE_NOISE=${USE_NOISE:-1}

for MODEL in LSTM TCN CNN_LSTM; do
  for PRED_LEN in 3 6 9 12; do
    python run.py \
      --model "${MODEL}" \
      --root_path ./data \
      --data_path "${DATA_PATH}" \
      --target "${TARGET}" \
      --seq_len 168 \
      --label_len 24 \
      --pred_len "${PRED_LEN}" \
      --num_variables 12 \
      --num_components 9 \
      --individual 1 \
      --use_noise "${USE_NOISE}" \
      --d_model 256 \
      --n_heads 4 \
      --dropout 0.2 \
      --batch_size 128 \
      --learning_rate 0.001 \
      --patience 10 \
      --gpu "${GPU}"
  done
done

# Ablation example:
# USE_NOISE=0 DATA_PATH=zhongshan_10m.csv TARGET=wind_speed_10m bash scripts/run_experiments.sh
#
# 100 m example:
# DATA_PATH=zhongshan_100m.csv TARGET=wind_speed_100m bash scripts/run_experiments.sh
