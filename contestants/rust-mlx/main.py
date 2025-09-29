import os
from pathlib import Path
import subprocess
import numpy as np
import argparse
from mlx_embeddings.utils import load
import mlx.core as mlx


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
    args = parser.parse_args()

    if args.device == "cpu":
        device = mlx.Device(mlx.cpu)
    elif args.device == "metal":
        device = mlx.Device(mlx.DeviceType.gpu)
    else:
        raise ValueError(f"mlx supports cpu and metal, got {args.device}")
    mlx.set_default_device(device)

    model, tokenizer = load(translate_model_name(args.model))

    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    inputs = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"input_ids {input_ids.shape} {input_ids}")
    print(f"attention_mask {attention_mask.shape} {attention_mask}")

    embedding_output = model.embeddings(input_ids)
    print(f"First layer: embeddings.\n {embedding_output}")

    am = model.get_extended_attention_mask(attention_mask)
    encoder_output = model.encoder(embedding_output, am)
    print(f"Encoder output\n {encoder_output}")

    embeddings = model(input_ids, attention_mask=attention_mask)

    print(embeddings.text_embeds)


if __name__ == "__main__":
    main()
