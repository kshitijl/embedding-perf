#!/usr/bin/env bash

for n in 32 64 128 256; do
  for device in metal cpu; do
    for bs in 8 32 64 128; do
      uv run main.py --model all-MiniLM-L6-v2 --max-seq-length $n --device $device --batch-size $bs --num-runs 5
    done
  done
done
