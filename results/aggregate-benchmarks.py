import json
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def discover_benchmark_files() -> List[Path]:
    """Discover all benchmark_results.jsonl files in contestant directories."""
    benchmark_files = []
    contestants_dir = Path("contestants")

    if not contestants_dir.exists():
        print("not exist")
        return benchmark_files

    for contestant_dir in contestants_dir.iterdir():
        print(f"contestant dir {contestant_dir}")
        if contestant_dir.is_dir():
            benchmark_file = contestant_dir / "output/benchmark_results.jsonl"
            if benchmark_file.exists():
                print("bchmark file exist")
                benchmark_files.append(benchmark_file)

    return benchmark_files


def parse_benchmark_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL benchmark results file."""
    results = []

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract run_times and filter out first 2 entries
                run_times = data.get("run_times", [])
                if len(run_times) < 2:
                    print(
                        f"Skipping line {line_num} in {file_path}: insufficient run_times ({len(run_times)})"
                    )
                    continue

                filtered_run_times = run_times[2:]  # Remove first 2 entries
                if len(filtered_run_times) == 0:
                    print(
                        f"Skipping line {line_num} in {file_path}: no run_times left after filtering"
                    )
                    continue

                # Create processed result
                result = {
                    "contestant": data.get("contestant"),
                    "language": data.get("language"),
                    "model": data.get("model"),
                    "device": data.get("device"),
                    "runtime": data.get("runtime"),
                    "max_seq_length": data.get("max_seq_length"),
                    "batch_size": data.get("batch_size"),
                    "total_time": data.get("total_time"),
                    "run_times": filtered_run_times,
                    "mean_runtime": np.mean(filtered_run_times),
                    "median_runtime": np.median(filtered_run_times),
                    "std_runtime": np.std(filtered_run_times),
                    "min_runtime": np.min(filtered_run_times),
                    "max_runtime": np.max(filtered_run_times),
                }

                results.append(result)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num} in {file_path}: {e}")
                continue

    return results


def load_all_benchmark_data() -> List[Dict[str, Any]]:
    """Load and aggregate all benchmark data."""
    all_results = []
    benchmark_files = discover_benchmark_files()

    print(f"Found {len(benchmark_files)} benchmark files:")
    for file_path in benchmark_files:
        print(f"  {file_path}")
        results = parse_benchmark_file(file_path)
        all_results.extend(results)
        print(f"    Loaded {len(results)} valid entries")

    print(f"\nTotal entries loaded: {len(all_results)}")
    return all_results


def create_performance_plots(results: List[Dict[str, Any]], output_dir="results"):
    """Create comprehensive performance plots in a single PDF."""
    if not results:
        print("No results to plot")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")

    # Create contestant x device combinations
    df['contestant_device'] = df['contestant'] + ' (' + df['device'] + ')'
    combinations = df['contestant_device'].unique()
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(combinations)))

    # Create PDF with all plots
    pdf_path = f"{output_dir}/benchmark_analysis.pdf"
    with PdfPages(pdf_path) as pdf:

        # Page 1: Overview plots (batch size and sequence length)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Benchmark Performance Overview", fontsize=16)

        # Plot 1: Performance by batch size (contestant x device combinations)
        ax1 = axes[0]
        for i, combo in enumerate(combinations):
            combo_data = df[df['contestant_device'] == combo]
            if len(combo_data) > 0:
                # Group by batch size and calculate mean
                batch_grouped = combo_data.groupby("batch_size")["mean_runtime"].mean()
                ax1.plot(
                    batch_grouped.index,
                    batch_grouped.values,
                    marker="o",
                    label=combo,
                    color=colors[i],
                )

        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Mean Runtime (seconds)")
        ax1.set_title("Performance by Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance by sequence length
        ax2 = axes[1]
        for i, combo in enumerate(combinations):
            combo_data = df[df['contestant_device'] == combo]
            if len(combo_data) > 0:
                # Group by sequence length and calculate mean
                seq_grouped = combo_data.groupby("max_seq_length")["mean_runtime"].mean()
                ax2.plot(
                    seq_grouped.index,
                    seq_grouped.values,
                    marker="o",
                    label=combo,
                    color=colors[i],
                )

        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Mean Runtime (seconds)")
        ax2.set_title("Performance by Sequence Length")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2+: Detailed batch size performance for each sequence length
        seq_lengths = sorted(df['max_seq_length'].unique())

        # Determine subplot layout
        n_seq_lengths = len(seq_lengths)
        if n_seq_lengths <= 3:
            cols = n_seq_lengths
            rows = 1
        else:
            cols = 3
            rows = (n_seq_lengths + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if n_seq_lengths == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("Batch Size Performance by Contestant (by Sequence Length)", fontsize=16)

        for plot_idx, seq_len in enumerate(seq_lengths):
            ax = axes[plot_idx]
            seq_len_data = df[df['max_seq_length'] == seq_len]

            contestant_devices = seq_len_data['contestant_device'].unique()
            contestant_device_colors = plt.colormaps["tab10"](np.linspace(0, 1, len(contestant_devices)))

            for i, contestant_device in enumerate(contestant_devices):
                contestant_device_data = seq_len_data[seq_len_data['contestant_device'] == contestant_device]
                if len(contestant_device_data) > 0:
                    # Group by batch size and calculate mean
                    batch_grouped = contestant_device_data.groupby("batch_size")["mean_runtime"].mean()
                    ax.plot(
                        batch_grouped.index,
                        batch_grouped.values,
                        marker="o",
                        label=contestant_device,
                        color=contestant_device_colors[i],
                    )

            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Mean Runtime (seconds)")
            ax.set_title(f"Seq Length = {seq_len}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots if any
        for plot_idx in range(n_seq_lengths, len(axes)):
            axes[plot_idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"All performance plots saved to {pdf_path}")


def generate_summary_report(
    results: List[Dict[str, Any]], output_file="results/benchmark_summary.txt"
):
    """Generate a comprehensive summary report."""
    if not results:
        print("No results to summarize")
        return

    df = pd.DataFrame(results)

    with open(output_file, "w") as f:
        f.write("BENCHMARK PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Create contestant_device combination
        df['contestant_device'] = df['contestant'] + ' (' + df['device'] + ')'

        # Overall statistics
        f.write(f"Total configurations tested: {len(results)}\n")
        f.write(f"Contestant-Device combinations: {', '.join(sorted(df['contestant_device'].unique()))}\n")
        f.write(f"Languages: {', '.join(sorted(df['language'].unique()))}\n")
        f.write(f"Models: {', '.join(sorted(df['model'].unique()))}\n")
        f.write(f"Devices: {', '.join(sorted(df['device'].unique()))}\n\n")

        # Best performers overall
        f.write("FASTEST OVERALL PERFORMANCES:\n")
        f.write("-" * 30 + "\n")
        fastest = df.nsmallest(10, "mean_runtime")
        for _, row in fastest.iterrows():
            f.write(
                f"{row['contestant_device']}/{row['language']} - {row['model']} - "
                f"batch_size={row['batch_size']} - {row['mean_runtime']:.3f}s\n"
            )
        f.write("\n")

        # Performance by contestant-device
        f.write("PERFORMANCE BY CONTESTANT-DEVICE:\n")
        f.write("-" * 30 + "\n")
        contestant_device_stats = df.groupby("contestant_device")["mean_runtime"].agg(
            ["mean", "std", "min", "max", "count"]
        )
        contestant_device_stats = contestant_device_stats.sort_values("mean")

        for contestant_device, stats in contestant_device_stats.iterrows():
            f.write(f"{contestant_device}:\n")
            f.write(f"  Average: {stats['mean']:.3f}s ± {stats['std']:.3f}s\n")
            f.write(f"  Best: {stats['min']:.3f}s\n")
            f.write(f"  Worst: {stats['max']:.3f}s\n")
            f.write(f"  Configurations: {stats['count']}\n\n")

        # Performance by language
        f.write("PERFORMANCE BY LANGUAGE:\n")
        f.write("-" * 30 + "\n")
        lang_stats = df.groupby("language")["mean_runtime"].agg(
            ["mean", "std", "min", "max", "count"]
        )
        lang_stats = lang_stats.sort_values("mean")

        for language, stats in lang_stats.iterrows():
            f.write(f"{language}:\n")
            f.write(f"  Average: {stats['mean']:.3f}s ± {stats['std']:.3f}s\n")
            f.write(f"  Best: {stats['min']:.3f}s\n")
            f.write(f"  Worst: {stats['max']:.3f}s\n")
            f.write(f"  Configurations: {stats['count']}\n\n")

        # Performance by batch size
        f.write("PERFORMANCE BY BATCH SIZE:\n")
        f.write("-" * 30 + "\n")
        batch_stats = df.groupby("batch_size")["mean_runtime"].agg(
            ["mean", "std", "min", "max", "count"]
        )
        batch_stats = batch_stats.sort_index()

        for batch_size, stats in batch_stats.iterrows():
            f.write(f"Batch size {batch_size}:\n")
            f.write(f"  Average: {stats['mean']:.3f}s ± {stats['std']:.3f}s\n")
            f.write(f"  Best: {stats['min']:.3f}s\n")
            f.write(f"  Worst: {stats['max']:.3f}s\n")
            f.write(f"  Configurations: {stats['count']}\n\n")

    print(f"Summary report saved to {output_file}")


def save_aggregated_data(
    results: List[Dict[str, Any]], output_file="results/aggregated_benchmarks.json"
):
    """Save aggregated data to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Aggregated data saved to {output_file}")


def main():
    # Change to repo root
    repo_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    os.chdir(repo_root)

    print("Starting benchmark aggregation...")

    # Load all benchmark data
    results = load_all_benchmark_data()

    if not results:
        print("No benchmark data found. Check your directory structure.")
        return

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Save aggregated data
    save_aggregated_data(results)

    # Generate plots
    create_performance_plots(results)

    # Generate summary report
    generate_summary_report(results)

    print(f"\nBenchmark aggregation complete! Processed {len(results)} configurations.")


if __name__ == "__main__":
    main()
