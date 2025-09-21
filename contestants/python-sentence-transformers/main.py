import os
from pathlib import Path
import subprocess
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import argparse
from typing import List
import time
import json


def embed(
    model: SentenceTransformer,
    sentences: List[str],
    device: str,
    max_seq_length: int,
    batch_size: int,
) -> np.ndarray:
    model.max_seq_length = max_seq_length  # type:ignore
    print(f"max seq len is {model.max_seq_length}")
    embeddings = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=True, device=device
    )
    return embeddings


def write_embeddings(embeddings: np.ndarray, embeddings_outfile: Path):
    with open(embeddings_outfile, "w") as f:
        for embedding in embeddings:
            text_line = np.array2string(
                embedding, max_line_width=10000000, separator=" "
            )[1:-1].strip()
            f.write(text_line + "\n")


def main():
    os.chdir(Path(__file__).parent)
    repo_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", default=Path(repo_root) / "data/sentences.txt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=1)
    args = parser.parse_args()

    if args.num_runs < 1:
        raise ValueError(f"num runs must be at least 1, given {args.num_runs}")

    print(
        f"Benchmarking {args.model} on {args.device}, seq len {args.max_seq_length}, batch size {args.batch_size}"
    )

    print("MKL:", torch.backends.mkl.is_available())
    print("OpenMP:", torch.backends.openmp.is_available())
    print(
        f"Threads: {torch.get_num_threads()}, interop threads: {torch.get_num_interop_threads()}"
    )

    start_total = time.time()
    model = SentenceTransformer(args.model, device=args.device)

    outfile = Path(
        f"output/{args.model}/{args.device}/embeddings-{args.max_seq_length}.txt"
    )
    benchmark_outfile = Path("output/benchmark_results.jsonl")
    os.makedirs(outfile.parent, exist_ok=True)

    sentences = open(args.sentences).readlines()

    run_times = []
    embeddings = None

    for i in range(args.num_runs):
        print(f"Run {i + 1} of {args.num_runs}")
        start_run = time.time()
        embeddings = embed(
            model,
            sentences,
            args.device,
            args.max_seq_length,
            args.batch_size,
        )
        end_run = time.time()
        run_times.append(end_run - start_run)

    end_total = time.time()
    total_time = end_total - start_total

    benchmark_result = {
        "model": args.model,
        "device": args.device,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "total_time": total_time,
        "run_times": run_times,
    }

    with open(benchmark_outfile, "a") as f:
        f.write(json.dumps(benchmark_result) + "\n")

    assert embeddings is not None
    write_embeddings(embeddings, outfile)


if __name__ == "__main__":
    main()
