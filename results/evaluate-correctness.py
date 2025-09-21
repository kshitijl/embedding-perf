import subprocess
import os
import json
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from collections import defaultdict


def load_embeddings(file_path: str):
    """Load embeddings from a text file with space-separated floats."""
    embeddings = []
    with open(file_path, "r") as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split()]
            embeddings.append(np.array(embedding))
    return np.array(embeddings)


def compute_embedding_errors(
    reference_embeddings: np.ndarray, contestant_embeddings: np.ndarray
) -> np.ndarray:
    """Compute per-vector max absolute differences."""
    if reference_embeddings.shape != contestant_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference_embeddings.shape} vs contestant {contestant_embeddings.shape}"
        )

    # Compute absolute differences for each vector
    abs_diffs = np.abs(reference_embeddings - contestant_embeddings)
    # Get max absolute difference for each vector
    max_abs_diffs = np.max(abs_diffs, axis=1)

    return max_abs_diffs


def discover_contestants() -> List[str]:
    """Discover all contestants in the contestants directory."""
    contestants_dir = Path("contestants")
    if not contestants_dir.exists():
        return []

    contestants = []
    for item in contestants_dir.iterdir():
        if item.is_dir():
            contestants.append(item.name)
    return contestants


def find_contestant_embeddings(contestant_name: str) -> List[dict]:
    """Find all embedding files for a contestant following the structure:
    contestants/{contestant}/output/{model}/{device}/embeddings-{seq_length}.txt
    """
    contestant_dir = Path("contestants") / contestant_name
    embeddings_files = []

    # Look for output directory
    output_dir = contestant_dir / "output"
    if not output_dir.exists():
        return embeddings_files

    # Iterate through model directories
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Iterate through device directories
        for device_dir in model_dir.iterdir():
            if not device_dir.is_dir():
                continue
            device_name = device_dir.name

            # Find embedding files
            for embedding_file in device_dir.glob("embeddings-*.txt"):
                # Extract sequence length and batch size from filename
                filename = embedding_file.name
                # Remove prefix and suffix
                name_part = filename.replace("embeddings-", "").replace(".txt", "")

                # Parse seq_length and batch_size
                if "-" in name_part:
                    # New format: embeddings-{seqlen}-{batchsize}.txt
                    parts = name_part.split("-")
                    seq_length = int(parts[0])
                    batch_size = int(parts[1])
                else:
                    # Legacy format: embeddings-{seqlen}.txt
                    seq_length = int(name_part)
                    batch_size = "unknown"

                embeddings_files.append(
                    {
                        "contestant": contestant_name,
                        "model": model_name,
                        "device": device_name,
                        "seq_length": seq_length,
                        "batch_size": batch_size,
                        "file_path": embedding_file,
                    }
                )

    return embeddings_files


def find_reference_embedding(model_name: str, seq_length: int) -> str | None:
    """Find the reference embedding file for a given model and sequence length."""
    reference_path = (
        Path("data/reference-output") / model_name / f"embeddings-{seq_length}.txt"
    )
    if reference_path.exists():
        return reference_path
    return None


def evaluate_all_contestants() -> List[dict]:
    """Evaluate all contestants and return results."""
    contestants = discover_contestants()
    results = []

    print(f"Found contestants: {contestants}")

    for contestant in contestants:
        print(f"\nEvaluating contestant: {contestant}")
        embedding_files = find_contestant_embeddings(contestant)

        for embedding_info in embedding_files:
            print(
                f"  Processing {embedding_info['model']} on {embedding_info['device']} with seq_length {embedding_info['seq_length']} batch_size {embedding_info['batch_size']}"
            )

            # Find corresponding reference
            reference_path = find_reference_embedding(
                embedding_info["model"], embedding_info["seq_length"]
            )
            if not reference_path:
                print(
                    f"    Warning: No reference found for {embedding_info['model']} seq_length {embedding_info['seq_length']}"
                )
                continue

            try:
                # Load embeddings
                reference_embeddings = load_embeddings(reference_path)
                contestant_embeddings = load_embeddings(embedding_info["file_path"])

                # Compute errors
                max_abs_diffs = compute_embedding_errors(
                    reference_embeddings, contestant_embeddings
                )

                # Compute statistics
                stats = {
                    "contestant": embedding_info["contestant"],
                    "model": embedding_info["model"],
                    "device": embedding_info["device"],
                    "seq_length": embedding_info["seq_length"],
                    "batch_size": embedding_info["batch_size"],
                    "mean_error": float(np.mean(max_abs_diffs)),
                    "median_error": float(np.median(max_abs_diffs)),
                    "max_error": float(np.max(max_abs_diffs)),
                    "min_error": float(np.min(max_abs_diffs)),
                    "std_error": float(np.std(max_abs_diffs)),
                    "num_vectors": len(max_abs_diffs),
                }

                results.append(stats)
                print(
                    f"    Mean error: {stats['mean_error']:.2e}, Max error: {stats['max_error']:.2e}"
                )

            except Exception as e:
                print(f"    Error processing: {e}")

    return results


def save_results(results: List[dict], output_file="results/evaluation_results.json"):
    """Save results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def plot_results(results: List[dict], output_file="results/evaluation_plot.png"):
    """Create plots showing the evaluation results."""
    if not results:
        print("No results to plot")
        return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Embedding Evaluation Results", fontsize=16)

    # Convert to DataFrame-like structure for easier plotting
    data = defaultdict(list)
    for result in results:
        key = f"{result['contestant']}-{result['device']}"
        data["contestant_device"].append(key)
        data["model"].append(result["model"])
        data["seq_length"].append(result["seq_length"])
        data["mean_error"].append(result["mean_error"])
        data["max_error"].append(result["max_error"])
        data["median_error"].append(result["median_error"])

    # Plot 1: Mean error by contestant and model
    ax1 = axes[0, 0]
    contestants = list(set(data["contestant_device"]))

    for i, contestant in enumerate(contestants):
        contestant_data = [
            r for r in results if f"{r['contestant']}-{r['device']}" == contestant
        ]
        errors_for_contestant = [r["mean_error"] for r in contestant_data]
        ax1.scatter(
            [i] * len(errors_for_contestant),
            errors_for_contestant,
            label=contestant,
            alpha=0.7,
            s=50,
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Contestant")
    ax1.set_ylabel("Mean Error (log scale)")
    ax1.set_title("Mean Error by Contestant")
    ax1.set_xticks(range(len(contestants)))
    ax1.set_xticklabels(contestants, rotation=45)
    ax1.legend()

    # Plot 2: Error by sequence length
    ax2 = axes[0, 1]
    seq_lengths = sorted(set(data["seq_length"]))
    for contestant in contestants:
        contestant_results = [
            r for r in results if f"{r['contestant']}-{r['device']}" == contestant
        ]
        seq_len_data = defaultdict(list)
        for r in contestant_results:
            seq_len_data[r["seq_length"]].append(r["mean_error"])

        seq_lens = []
        mean_errors = []
        for seq_len in seq_lengths:
            if seq_len in seq_len_data:
                seq_lens.append(seq_len)
                mean_errors.append(np.mean(seq_len_data[seq_len]))

        if seq_lens:
            ax2.plot(seq_lens, mean_errors, marker="o", label=contestant)

    ax2.set_yscale("log")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Mean Error (log scale)")
    ax2.set_title("Mean Error by Sequence Length")
    ax2.legend()

    # Plot 3: Error distribution (box plot)
    ax3 = axes[1, 0]
    error_data_by_contestant = defaultdict(list)
    for result in results:
        key = f"{result['contestant']}-{result['device']}"
        error_data_by_contestant[key].append(result["mean_error"])

    ax3.boxplot(
        [error_data_by_contestant[c] for c in contestants], tick_labels=contestants
    )
    ax3.set_yscale("log")
    ax3.set_ylabel("Mean Error (log scale)")
    ax3.set_title("Error Distribution by Contestant")
    ax3.tick_params(axis="x", rotation=45)

    # Plot 4: Max vs Mean error
    ax4 = axes[1, 1]
    for contestant in contestants:
        contestant_results = [
            r for r in results if f"{r['contestant']}-{r['device']}" == contestant
        ]
        mean_errors = [r["mean_error"] for r in contestant_results]
        max_errors = [r["max_error"] for r in contestant_results]
        ax4.scatter(mean_errors, max_errors, label=contestant, alpha=0.7, s=50)

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Mean Error (log scale)")
    ax4.set_ylabel("Max Error (log scale)")
    ax4.set_title("Max vs Mean Error")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def main():
    repo_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    os.chdir(repo_root)

    print("Starting contestant evaluation...")

    # Evaluate all contestants
    results = evaluate_all_contestants()

    if not results:
        print("No results found. Check your directory structure.")
        return

    # Save results
    save_results(results)

    # Create plots
    plot_results(results)

    # Print summary
    print(f"\nEvaluation complete! Processed {len(results)} configurations.")

    # Print top-level summary
    contestants = set(r["contestant"] for r in results)
    for contestant in contestants:
        contestant_results = [r for r in results if r["contestant"] == contestant]
        max_errors = [r["max_error"] for r in contestant_results]
        print(f"\n{contestant}:")
        print(f"  Average max error: {np.max(max_errors):.2e}")
        print(f"  Best max error: {np.min(max_errors):.2e}")
        print(f"  Worst max error: {np.max(max_errors):.2e}")


if __name__ == "__main__":
    main()
