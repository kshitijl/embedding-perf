#!/usr/bin/env bash

for n in 32 64 128 256; do
  for device in mps cpu; do
    for bs in 8 32 64 128; do
      echo cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length $n --device $device --embedding-dim 384 --batch-size $bs --num-runs 13
    done
  done
done
