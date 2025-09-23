use anyhow::{Error as E, Result};
use candle_core::{Device as CandleDevice, IndexOp, Tensor};
use candle_transformers::models::bert::{BertModel, Config, DTYPE as BERT_DTYPE};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api as HfApi;
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

struct EmbedderModel {
    model: BertModel,
    device: CandleDevice,
}

impl EmbedderModel {
    fn load(model_name: &str, device: Device) -> Result<Self> {
        let candle_device = match device {
            Device::Cpu => CandleDevice::Cpu,
            Device::Mps => CandleDevice::new_metal(0)?,
        };

        let api = HfApi::new()?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
        let config_filename = repo.get("config.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                BERT_DTYPE,
                &candle_device,
            )?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(EmbedderModel {
            model,
            device: candle_device,
        })
    }

    fn embed(
        &self,
        input_ids: &[Vec<i64>],
        attention_masks: &[Vec<i64>],
        processing_batch_size: usize,
    ) -> Result<Tensor> {
        if input_ids.is_empty() {
            panic!("empty input");
        }

        let total_samples = input_ids.len();
        let seq_length = input_ids[0].len();

        for (ids, mask) in input_ids.iter().zip(attention_masks.iter()) {
            if ids.len() != seq_length || mask.len() != seq_length {
                return Err(E::msg("All sequences must have same length"));
            }
        }

        let flat_input_ids: Vec<i64> = input_ids.iter().flatten().copied().collect();
        let flat_attention_mask: Vec<i64> = attention_masks.iter().flatten().copied().collect();

        let input_ids_tensor =
            Tensor::new(flat_input_ids.as_slice(), &self.device)?.reshape((input_ids.len(), ()))?;
        let attention_masks_tensor = Tensor::new(flat_attention_mask.as_slice(), &self.device)?
            .reshape((input_ids.len(), ()))?;
        let token_type_ids_tensor = Tensor::zeros_like(&input_ids_tensor)?;

        let num_batches = (total_samples + processing_batch_size - 1) / processing_batch_size;
        let mut all_batch_embeddings = Vec::new();
        all_batch_embeddings.reserve_exact(num_batches);

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * processing_batch_size;
            let batch_end = usize::min(batch_start + processing_batch_size, total_samples);

            eprintln!(
                "Processing batch {} of {}, from {} to {}",
                batch_idx, num_batches, batch_start, batch_end
            );

            let batch_input_ids = input_ids_tensor.i((batch_start..batch_end, ..))?;
            let batch_attention_masks = attention_masks_tensor.i((batch_start..batch_end, ..))?;
            let batch_token_type_ids = token_type_ids_tensor.i((batch_start..batch_end, ..))?;

            let result = self.model.forward(
                &batch_input_ids,
                &batch_token_type_ids,
                Some(&batch_attention_masks),
            )?;

            all_batch_embeddings.push(result.to_device(&CandleDevice::Cpu)?.contiguous()?);
        }

        let unpooled = Tensor::cat(&all_batch_embeddings, 0)?;
        let attention_mask_3d = attention_masks_tensor.unsqueeze(2)?.to_dtype(BERT_DTYPE)?;
        let embeddings = (unpooled.broadcast_mul(&attention_mask_3d)?).sum(1)?;
        let sum_mask = attention_mask_3d.sum(1)?;
        let embeddings = embeddings.broadcast_div(&sum_mask)?;

        let normalized = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        let answer = normalized.to_device(&CandleDevice::Cpu)?.contiguous()?;

        Ok(answer)
    }
}

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
    num_tokens: usize,
}

#[derive(Debug)]
struct TokenBatch {
    input_ids: Vec<Vec<i64>>,
    attention_mask: Vec<Vec<i64>>,
}

impl TokenBatch {
    fn load(path: &Path, max_length: usize) -> Result<Self> {
        #[derive(Debug, Deserialize)]
        struct TokenizedInput {
            input_ids: Vec<Vec<i64>>,
            attention_mask: Vec<Vec<i64>>,
        }
        let content = fs::read_to_string(path).map_err(E::msg)?;

        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();

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

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_masks);
        }

        Ok(Self {
            input_ids: all_input_ids,
            attention_mask: all_attention_masks,
        })
    }
}

fn write_embeddings(path: &Path, embeddings: &Tensor) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (nvecs, embedding_dim) = embeddings.dims2()?;
    for vec_idx in 0..nvecs {
        for val_idx in 0..embedding_dim {
            let val = embeddings
                .i((vec_idx, val_idx))
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
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
    let model = EmbedderModel::load(&model_name, args.device)?;
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

    // Calculate total tokens for one full run (all sequences)
    let num_tokens = input
        .input_ids
        .iter()
        .map(|seq| seq.iter().filter(|&&token| token != 0).count()) // Count non-padding tokens
        .sum();

    let benchmark_result = BenchmarkResult {
        language: "rust".to_string(),
        os: "macos".to_string(),
        contestant: "rust-candle".to_string(),
        model: args.model.clone(),
        device: args.device.to_string(),
        max_seq_length: args.max_seq_length,
        batch_size: args.batch_size,
        total_time,
        run_times,
        num_tokens,
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
