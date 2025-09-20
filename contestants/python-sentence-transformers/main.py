import os
from pathlib import Path
import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from typing import List


def embed(
    sentences: List[str], model_name: str, device: str, max_seq_length: int
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length  # type:ignore
    print(f"max seq len is {model.max_seq_length}")
    embeddings = model.encode(sentences, show_progress_bar=True)
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
    args = parser.parse_args()

    outfile = Path(
        f"output/{args.model}/{args.device}/embeddings-{args.max_seq_length}.txt"
    )
    os.makedirs(outfile.parent, exist_ok=True)

    sentences = open(args.sentences).readlines()
    embeddings = embed(sentences, args.model, args.device, args.max_seq_length)
    write_embeddings(embeddings, outfile)


if __name__ == "__main__":
    main()
