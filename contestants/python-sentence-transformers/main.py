from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from typing import List


def embed(sentences: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings


def write_embeddings(embeddings: np.ndarray, embeddings_outfile: str):
    with open(embeddings_outfile, "w") as f:
        for embedding in embeddings:
            text_line = np.array2string(
                embedding, max_line_width=10000000, separator=" "
            )[1:-1].strip()
            f.write(text_line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences")
    parser.add_argument("model")
    parser.add_argument("embeddings_outfile")
    # parser.add_argument("timings-output", required=True)
    args = parser.parse_args()

    sentences = open(args.sentences).readlines()
    embeddings = embed(sentences, args.model)
    write_embeddings(embeddings, args.embeddings_outfile)


if __name__ == "__main__":
    main()
