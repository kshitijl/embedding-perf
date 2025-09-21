#!/usr/bin/env bash

cmake --build build --config Release

for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      build/main --model all-MiniLM-L6-v2 --max-seq-length $n --device mps --embedding-dim 384 --batch-size $bs --num-runs 5
  done
done


for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      DYLD_LIBRARY_PATH=$(git rev-parse --show-toplevel)/.venv/lib/python3.12/site-packages/torch/lib:$DYLD_LIBRARY_PATH build/main --model all-MiniLM-L6-v2 --max-seq-length $n --device cpu --embedding-dim 384 --batch-size $bs --num-runs 5
  done
done

for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      DYLD_LIBRARY_PATH=/opt/homebrew/opt/pytorch/lib:$DYLD_LIBRARY_PATH build/main --model all-MiniLM-L6-v2 --max-seq-length $n --device cpu --embedding-dim 384 --batch-size $bs --num-runs 5
  done
done
