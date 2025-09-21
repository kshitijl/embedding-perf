#!/usr/bin/env bash

for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      uv run main.py --model ../../onnx-models/all-MiniLM-L6-v2 --max-seq-length $n --device cpu --batch-size $bs --num-runs 13
    done
done
