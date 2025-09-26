import os
from pathlib import Path
import subprocess
import numpy as np
import argparse
import time
import json
from mlx.nn import Module as MlxModule
from mlx_embeddings.utils import load
from tqdm import tqdm
import mlx.core as mlx


def embed_from_tokens(
    model: MlxModule,
    input_ids,
    attention_mask,
    batch_size: int,
) -> np.ndarray:
    all_embeddings = []

    num_samples = len(input_ids)

    pbar = tqdm(total=num_samples, desc="Embedding sentences", unit="sentence")

    for i in range(0, num_samples, batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]
        all_embeddings.append(
            np.array(
                model(batch_input_ids, attention_mask=batch_attention_mask).text_embeds
            )
        )

        pbar.update(len(batch_input_ids))

    pbar.close()
    return np.vstack(all_embeddings)


def write_embeddings(embeddings: np.ndarray, embeddings_outfile: Path):
    """
    Writes embeddings to a text file, with each line representing an embedding.
    """
    print(f"Writing {len(embeddings)} embeddings to {embeddings_outfile}...")
    with open(embeddings_outfile, "w") as f:
        for embedding in embeddings:
            text_line = np.array2string(
                embedding, max_line_width=10000000, separator=" "
            )[1:-1].strip()
            f.write(text_line + "\n")
    print("Done.")


def translate_model_name(name: str) -> str:
    d = {"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"}
    return d[name]


def main():
    os.chdir(Path(__file__).parent)
    try:
        repo_root = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        print("Not a git repository. Using current directory as repo root.")
        repo_root = "."
    repo_root = Path(repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--device",
        help="Device to run on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--sentences", type=str, default=repo_root / "data/sentences.txt"
    )
    args = parser.parse_args()

    if args.num_runs < 1:
        raise ValueError(f"num runs must be at least 1, given {args.num_runs}")

    if args.device == "cpu":
        device = mlx.Device(mlx.cpu)
    elif args.device == "metal":
        device = mlx.Device(mlx.DeviceType.gpu)
    else:
        raise ValueError(f"mlx supports cpu and metal, got {args.device}")
    mlx.set_default_device(device)

    print(
        f"Benchmarking mlx model from {args.model} on {args.device}, seq len {args.max_seq_length}, batch size {args.batch_size}"
    )

    start_total = time.time()
    model, tokenizer = load(translate_model_name(args.model))
    sentences = list(open(args.sentences).readlines())
    inputs = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
    )

    outfile = Path(
        f"output/{args.model}/{args.device}/embeddings-{args.max_seq_length}-{args.batch_size}.txt"
    )
    benchmark_outfile = Path("output/benchmark_results.jsonl")
    os.makedirs(outfile.parent, exist_ok=True)
    os.makedirs(benchmark_outfile.parent, exist_ok=True)

    run_times = []
    embeddings = None

    for i in range(args.num_runs):
        print(f"Run {i + 1} of {args.num_runs}")
        start_run = time.time()
        embeddings = embed_from_tokens(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            args.batch_size,
        )
        end_run = time.time()
        run_times.append(end_run - start_run)

    end_total = time.time()
    total_time = end_total - start_total

    benchmark_result = {
        "contestant": "mlx",
        "language": "python",
        "os": "macos",
        "model": args.model,
        "device": args.device,
        "runtime": "onnx",
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
