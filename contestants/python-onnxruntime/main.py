import os
from pathlib import Path
import subprocess
import onnxruntime as ort
import numpy as np
import argparse
from typing import List, Dict, Any
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


def load_tokenized_input(file_path: Path, max_seq_length: int) -> Dict[str, np.ndarray]:
    """
    Load pre-tokenized input from JSONL file, similar to the Rust implementation.
    """
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract arrays (they come nested as [[...]])
                input_ids = data['input_ids'][0]
                attention_mask = data['attention_mask'][0]
                token_type_ids = data.get('token_type_ids', [[]])[0] if data.get('token_type_ids') else [0] * len(input_ids)

                # Handle truncation and padding like Rust implementation
                if len(input_ids) > max_seq_length:
                    input_ids = input_ids[:max_seq_length - 1] + [102]  # Add SEP token
                    attention_mask = attention_mask[:max_seq_length - 1] + [1]
                    token_type_ids = token_type_ids[:max_seq_length - 1] + [0]

                # Pad to max_seq_length
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    attention_mask.append(0)
                    token_type_ids.append(0)

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_token_type_ids.append(token_type_ids)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    return {
        'input_ids': np.array(all_input_ids, dtype=np.int64),
        'attention_mask': np.array(all_attention_masks, dtype=np.int64),
        'token_type_ids': np.array(all_token_type_ids, dtype=np.int64)
    }


def embed_from_tokens(
    session: ort.InferenceSession,
    tokenized_input: Dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """
    Generates embeddings from pre-tokenized input using an ONNX model.
    """
    all_embeddings = []
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    token_type_ids = tokenized_input['token_type_ids']

    num_samples = len(input_ids)

    # Set up the progress bar
    pbar = tqdm(total=num_samples, desc="Embedding sentences", unit="sentence")

    for i in range(0, num_samples, batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        batch_token_type_ids = token_type_ids[i:i + batch_size]

        # Prepare the model inputs for ONNX Runtime
        model_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
        }

        # Run inference
        model_outputs = session.run(None, model_inputs)
        last_hidden_state = model_outputs[0]
        assert isinstance(last_hidden_state, np.ndarray)

        # Perform pooling and normalization
        pooled_embeddings = mean_pooling(last_hidden_state, batch_attention_mask)
        normalized_embeddings = normalize(pooled_embeddings)

        all_embeddings.append(normalized_embeddings)
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
        help="Model name (will look for tokenized input in data/reference-output/{model}/tokenized.txt)",
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
    tokenized_input_file = repo_root / f"data/reference-output/{args.model}/tokenized.txt"

    if not onnx_model_file.exists():
        raise FileNotFoundError(f"Could not find 'model.onnx' in {model_path}")

    if not tokenized_input_file.exists():
        raise FileNotFoundError(f"Could not find tokenized input at {tokenized_input_file}")

    start_total = time.time()

    # Load pre-tokenized input
    print("Loading pre-tokenized input...")
    tokenized_input = load_tokenized_input(tokenized_input_file, args.max_seq_length)
    print(f"Loaded {len(tokenized_input['input_ids'])} tokenized sequences")

    # Load ONNX model
    print("Loading ONNX inference session...")
    session = ort.InferenceSession(str(onnx_model_file), providers=execution_providers)
    print("Model loaded successfully.")

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
            session,
            tokenized_input,
            args.batch_size,
        )
        end_run = time.time()
        run_times.append(end_run - start_run)

    end_total = time.time()
    total_time = end_total - start_total

    # Calculate num_tokens (total non-padding tokens for one full run)
    num_tokens = int(np.sum(tokenized_input['attention_mask']))

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
        "num_tokens": num_tokens,
    }

    with open(benchmark_outfile, "a") as f:
        f.write(json.dumps(benchmark_result) + "\n")

    assert embeddings is not None
    write_embeddings(embeddings, outfile)


if __name__ == "__main__":
    main()
