#!/bin/bash

cat ../contestants/*/output/benchmark_results.jsonl > all_benchmarks.jsonl

echo "Built all_benchmarks.jsonl from $(cat ../contestants/*/output/benchmark_results.jsonl | wc -l) entries"
