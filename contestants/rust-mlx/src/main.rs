use anyhow::{Error as E, Result};
use bert::BertEmbedder;
use clap::{Parser, ValueEnum};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

mod bert;

#[derive(Serialize)]
struct BenchmarkResult {
    contestant: String,
    os: String,
    language: String,
    model: String,
    device: String,
    max_seq_length: usize,
    batch_size: usize,
    total_time: f64,
    run_times: Vec<f64>,
}

#[derive(Debug)]
struct TokenBatch {
    input_ids: Array,
    attention_mask: Array,
}

impl TokenBatch {
    fn load(path: &Path, max_length: usize) -> Result<Self> {
        #[derive(Debug, Deserialize)]
        struct TokenizedInput {
            input_ids: Vec<Vec<u32>>,
            attention_mask: Vec<Vec<u32>>,
        }
        let content = fs::read_to_string(path).map_err(E::msg)?;

        let mut all_input_ids: Vec<Array> = Vec::new();
        let mut all_attention_masks: Vec<Array> = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let input: TokenizedInput = serde_json::from_str(line)
                .map_err(|e| E::msg(format!("JSON parsing error: {}", e)))?;
            if input.input_ids.len() != 1 || input.attention_mask.len() != 1 {
                return Err(E::msg(format!(
                    "Expected all input to be array (batch size one), got {}",
                    input.input_ids.len()
                )));
            }
            let mut input_ids = input.input_ids.into_iter().next().unwrap();
            let mut attention_masks = input.attention_mask.into_iter().next().unwrap();

            if input_ids.len() > max_length {
                input_ids.truncate(max_length - 1);
                attention_masks.truncate(max_length - 1);

                // Add [SEP] token since we truncated
                // TODO this only works for BERT tokenizers
                input_ids.push(102);
                attention_masks.push(1);
            }

            input_ids.resize(max_length, 0);
            attention_masks.resize(max_length, 0);

            all_input_ids.push(Array::from_slice(&input_ids, &[max_length as i32]));
            all_attention_masks.push(Array::from_slice(&attention_masks, &[max_length as i32]));
        }

        let all_input_ids = mlx_rs::ops::stack(&all_input_ids)?;
        let all_attention_masks = mlx_rs::ops::stack(&all_attention_masks)?;

        Ok(Self {
            input_ids: all_input_ids,
            attention_mask: all_attention_masks,
        })
    }
}

fn write_embeddings(path: &Path, embeddings: &Array) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    assert_eq!(embeddings.ndim(), 2);
    let nvecs = embeddings.dim(0) as usize;
    let embedding_dim = embeddings.dim(1) as usize;
    let embeddings = embeddings.as_slice::<f32>();
    for vec_idx in 0..nvecs {
        for val_idx in 0..embedding_dim {
            let val = embeddings[vec_idx * embedding_dim + val_idx];
            if val >= 0.0 {
                write!(writer, "  {:.8e}", val)?;
            } else {
                write!(writer, " {:.8e}", val)?;
            }
        }
        write!(writer, "\n")?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Device {
    Cpu,
    Mps,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.to_possible_value().unwrap().get_name().to_string();
        write!(f, "{}", name)
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Model name
    #[arg(long)]
    model: String,

    /// Truncate each tokenized input to this many tokens
    #[arg(short, long)]
    max_seq_length: usize,

    /// Whether the model uses MPS. Traced TorchScript models are device-specific.
    #[arg(short, long)]
    device: Device,

    /// Batch size for processing
    #[arg(long)]
    batch_size: usize,

    /// Number of runs for benchmarking
    #[arg(long, default_value = "1")]
    num_runs: usize,
}

fn get_repo_root() -> PathBuf {
    let repo_root = Command::new("git")
        .args(&["rev-parse", "--show-toplevel"])
        .output()
        .unwrap()
        .stdout;
    PathBuf::from(String::from_utf8(repo_root).unwrap().trim())
}

fn get_output_path(args: &Args) -> PathBuf {
    let output_path = format!(
        "output/{}/{}/embeddings-{}-{}.txt",
        args.model, args.device, args.max_seq_length, args.batch_size
    );
    let parent = std::path::Path::new(&output_path).parent().unwrap();
    fs::create_dir_all(parent).unwrap();

    PathBuf::from(output_path)
}

fn get_input_tokens_path(args: &Args, repo_root: &Path) -> PathBuf {
    repo_root.join(format!(
        "data/reference-output/{}/tokenized.txt",
        args.model
    ))
}

fn get_hf_model_name(model_name: &str) -> &'static str {
    match model_name {
        "all-MiniLM-L6-v2" => "sentence-transformers/all-MiniLM-L6-v2",
        _ => panic!("unknown model {}", model_name),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.num_runs < 1 {
        return Err(E::msg(format!(
            "num runs must be at least 1, given {}",
            args.num_runs
        )));
    }

    let repo_root = get_repo_root();
    let output_path = get_output_path(&args);
    let input_tokens_path = get_input_tokens_path(&args, &repo_root);
    let model_name = get_hf_model_name(&args.model);

    let start_total = Instant::now();
    eprintln!("Loading model");

    let device = match args.device {
        Device::Cpu => mlx_rs::DeviceType::Cpu,
        Device::Mps => mlx_rs::DeviceType::Gpu,
    };
    mlx_rs::Device::set_default(&mlx_rs::Device::new(device, 0));

    let model = BertEmbedder::load(&model_name)?;
    eprintln!("Parsing input tokens");
    let input = TokenBatch::load(&input_tokens_path, args.max_seq_length)?;

    let mut run_times = Vec::new();
    let mut embeddings = None;

    for i in 0..args.num_runs {
        eprintln!("Run {}/{}", i + 1, args.num_runs);
        let start_run = Instant::now();
        embeddings = Some(model.embed(&input.input_ids, &input.attention_mask, args.batch_size)?);
        let end_run = Instant::now();
        run_times.push(end_run.duration_since(start_run).as_secs_f64());
    }

    let end_total = Instant::now();
    let total_time = end_total.duration_since(start_total).as_secs_f64();

    let benchmark_result = BenchmarkResult {
        language: "rust".to_string(),
        os: "macos".to_string(),
        contestant: "mlx-rs".to_string(),
        model: args.model.clone(),
        device: args.device.to_string(),
        max_seq_length: args.max_seq_length,
        batch_size: args.batch_size,
        total_time,
        run_times,
    };

    let benchmark_path = PathBuf::from("output/benchmark_results.jsonl");
    if let Some(parent) = benchmark_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(benchmark_path)?;
    writeln!(file, "{}", serde_json::to_string(&benchmark_result)?)?;

    eprintln!("Writing embeddings to file");
    write_embeddings(&output_path, &embeddings.unwrap())?;
    eprintln!("Done");
    Ok(())
}
