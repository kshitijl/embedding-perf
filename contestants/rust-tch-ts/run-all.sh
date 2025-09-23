#!/usr/bin/env bash

for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      DYLD_LIBRARY_PATH=~/Downloads/libtorch-2.7.0/lib:$DYLD_LIBRARY_PATH cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length $n --device mps --embedding-dim 384 --batch-size $bs --num-runs 5
  done
done


for n in 32 64 128 256; do
    for bs in 8 32 64 128; do
      DYLD_LIBRARY_PATH=~/Downloads/libtorch-2.7.0/lib:$DYLD_LIBRARY_PATH cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length $n --device cpu --embedding-dim 384 --batch-size $bs --num-runs 5
  done
done

# for n in 32 64 128 256; do
#     for bs in 8 32 64 128; do
#       DYLD_LIBRARY_PATH=/opt/homebrew/opt/pytorch/lib:$DYLD_LIBRARY_PATH cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length $n --device cpu --embedding-dim 384 --batch-size $bs --num-runs 5
#   done
# done
