from sentence_transformers import SentenceTransformer
import json
import os

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)


def tokenize_using(model_name: str, output_file: str):
    sentences = open(os.path.join(script_directory, "sentences.txt")).readlines()

    model = SentenceTransformer(model_name)

    with open(output_file, "w") as f:
        for sentence in sentences:
            tokens = model.tokenize([sentence])
            j = {}
            for key, tensor in tokens.items():
                j[key] = tensor.tolist()
            json_line = json.dumps(j, ensure_ascii=False)
            f.write(json_line + "\n")


def batch():
    batch_inputs = [
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L12-v2", "all-MiniLM-L12-v2"),
        ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
        ("intfloat/e5-base-v2", "e5-base-v2"),
        ("avsolatorio/GIST-Small-Embedding-v0", "GIST-Small-Embedding-v0"),
        ("thenlper/gte-large", "gte-large"),
    ]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for model, filename in batch_inputs:
        tokenize_using(model, os.path.join(output_dir, filename))


def main():
    batch()


if __name__ == "__main__":
    main()
