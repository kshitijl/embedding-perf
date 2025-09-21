import os
from pathlib import Path
import subprocess
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np
import argparse
from typing import List
import time
import json
from tqdm import tqdm


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Performs mean pooling on the token embeddings.

    Uses the attention mask to correctly average the token embeddings.
    """
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
        token_embeddings.shape[-1], axis=-1
    )
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Performs L2 normalization on the embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def embed(
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    sentences: List[str],
    batch_size: int,
) -> np.ndarray:
    """
    Generates embeddings for a list of sentences using an ONNX model.
    """
    all_embeddings = []
    num_sentences = len(sentences)

    # Set up the progress bar
    pbar = tqdm(total=num_sentences, desc="Embedding sentences", unit="sentence")

    for i in range(0, num_sentences, batch_size):
        batch_sentences = sentences[i : i + batch_size]

        # 1. Tokenize the input sentences
        encoded_input = tokenizer.encode_batch(batch_sentences)

        input_ids = np.array([e.ids for e in encoded_input], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encoded_input], dtype=np.int64
        )
        token_type_ids = np.array([e.type_ids for e in encoded_input], dtype=np.int64)

        # 2. Prepare the model inputs for ONNX Runtime
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # 3. Run inference
        model_outputs = session.run(None, model_inputs)
        last_hidden_state = model_outputs[0]
        assert isinstance(last_hidden_state, np.ndarray)

        # 4. Perform pooling and normalization
        pooled_embeddings = mean_pooling(last_hidden_state, attention_mask)
        normalized_embeddings = normalize(pooled_embeddings)

        all_embeddings.append(normalized_embeddings)
        pbar.update(len(batch_sentences))

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
    parser.add_argument("--sentences", default=Path(repo_root) / "data/sentences.txt")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the directory containing model.onnx and tokenizer.json",
    )
    parser.add_argument(
        "--device",
        help="Device to run on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=1)
    args = parser.parse_args()

    if args.num_runs < 1:
        raise ValueError(f"num runs must be at least 1, given {args.num_runs}")

    if args.device == "cpu":
        execution_providers = ["CPUExecutionProvider"]
    elif args.device == "coreml":
        execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        raise ValueError(
            f"Only cpu and coreml execution providers are supported right now, got {args.device}"
        )

    print(
        f"Benchmarking ONNX model from {args.model} on {args.device}, seq len {args.max_seq_length}, batch size {args.batch_size}"
    )

    model_path = repo_root / f"models/onnx/{args.model}"
    onnx_model_file = model_path / "model.onnx"
    tokenizer_file = model_path / "tokenizer.json"

    if not onnx_model_file.exists() or not tokenizer_file.exists():
        raise FileNotFoundError(
            f"Could not find 'model.onnx' and/or 'tokenizer.json' in {model_path}"
        )

    start_total = time.time()

    # Load tokenizer and configure it
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.enable_truncation(max_length=args.max_seq_length)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=args.max_seq_length)
    print(f"Tokenizer max sequence length set to {args.max_seq_length}")

    # Load ONNX model
    print("Loading ONNX inference session...")
    session = ort.InferenceSession(str(onnx_model_file), providers=execution_providers)
    print("Model loaded successfully.")

    model_name_for_path = model_path.name
    outfile = Path(
        f"output/{model_name_for_path}/{args.device}/embeddings-{args.max_seq_length}-{args.batch_size}.txt"
    )
    benchmark_outfile = Path("output/benchmark_results.jsonl")
    os.makedirs(outfile.parent, exist_ok=True)
    os.makedirs(benchmark_outfile.parent, exist_ok=True)

    with open(args.sentences) as f:
        sentences = [line.strip() for line in f.readlines()]

    run_times = []
    embeddings = None

    for i in range(args.num_runs):
        print(f"Run {i + 1} of {args.num_runs}")
        start_run = time.time()
        embeddings = embed(
            session,
            tokenizer,
            sentences,
            args.batch_size,
        )
        end_run = time.time()
        run_times.append(end_run - start_run)

    end_total = time.time()
    total_time = end_total - start_total

    benchmark_result = {
        "contestant": "onnxruntime",
        "language": "python",
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
