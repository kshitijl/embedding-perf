from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

TOKENIZATION_MAX_LENGTH = 256
TRUNCATION_SIZES_FOR_EMBEDDINGS = [32, 64, 128, 256]

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)


def tokenize_using(model_name: str, output_file: str):
    print(f"Generating tokens for {model_name} to {output_file}")
    sentences = open(os.path.join(script_directory, "sentences.txt")).readlines()

    model = SentenceTransformer(model_name)
    model.max_seq_length = TOKENIZATION_MAX_LENGTH  # type:ignore

    with open(output_file, "w") as f:
        for sentence in sentences:
            tokens = model.tokenize([sentence])
            j = {}
            for key, tensor in tokens.items():
                j[key] = tensor.tolist()
            json_line = json.dumps(j, ensure_ascii=False)
            f.write(json_line + "\n")


def generate_reference_embeddings(model_name: str, output_dir: str):
    print(f"Generating embeddings for {model_name} at {output_dir}")
    sentences = open(os.path.join(script_directory, "sentences.txt")).readlines()
    model = SentenceTransformer(model_name)

    batch_size = 64

    for max_length in TRUNCATION_SIZES_FOR_EMBEDDINGS:
        print(f"    max length {max_length}")
        model.max_seq_length = max_length

        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        output_file = os.path.join(output_dir, f"embeddings-{max_length}.txt")
        with open(output_file, "w") as f:
            for embedding in embeddings:
                text_line = np.array2string(
                    embedding, max_line_width=10000000, separator=" "
                )[1:-1].strip()
                f.write(text_line + "\n")
        print(f"    done, wrote to {output_file}")


def batch():
    batch_inputs = [
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L12-v2", "all-MiniLM-L12-v2"),
        ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
        ("intfloat/e5-base-v2", "e5-base-v2"),
        ("avsolatorio/GIST-Small-Embedding-v0", "GIST-Small-Embedding-v0"),
        ("thenlper/gte-large", "gte-large"),
    ]

    output_dir = "reference-output"
    os.makedirs(output_dir, exist_ok=True)

    for model, dir_name in batch_inputs:
        model_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(model_output_dir, exist_ok=True)
        tokenize_filename = os.path.join(model_output_dir, "tokenized.txt")
        tokenize_using(model, tokenize_filename)
        generate_reference_embeddings(model, model_output_dir)


def main():
    batch()


if __name__ == "__main__":
    main()
