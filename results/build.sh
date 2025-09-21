#!/bin/bash

# Simple build script to concatenate all benchmark JSONL files
cat ../contestants/*/output/benchmark_results.jsonl > all_benchmarks.jsonl

echo "Built all_benchmarks.jsonl from $(cat ../contestants/*/output/benchmark_results.jsonl | wc -l) entries"