use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_int, c_long};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

unsafe extern "C" {
    fn embedder_load_model(model_path: *const c_char, use_mps: c_int) -> *mut std::ffi::c_void;
    fn embedder_free_model(model_handle: *mut std::ffi::c_void);
    fn embedder_embed(
        model_handle: *mut std::ffi::c_void,
        input_ids: *const i64,
        attention_mask: *const i64,
        total_samples: c_long,
        seq_length: c_long,
        processing_batch_size: c_long,
        output_embeddings: *mut f32,
        output_buffer_size_in_bytes: c_long,
    ) -> c_long;
}

struct EmbedderModel {
    handle: *mut std::ffi::c_void,
    embedding_dim: usize,
}

impl EmbedderModel {
    fn load(model_path: &Path, device: Device, embedding_dim: usize) -> Result<Self> {
        let c_path = CString::new(model_path.to_string_lossy().as_ref()).map_err(E::msg)?;

        let use_mps = match device {
            Device::Mps => 1,
            Device::Cpu => 0,
        };

        let handle = unsafe { embedder_load_model(c_path.as_ptr(), use_mps) };

        if handle.is_null() {
            Err(E::msg("failed to load model"))
        } else {
            Ok(EmbedderModel {
                handle,
                embedding_dim,
            })
        }
    }

    fn embed_batch(
        &self,
        input_ids: &[Vec<i64>],
        attention_masks: &[Vec<i64>],
        processing_batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
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

        let output_size = total_samples * self.embedding_dim;
        let mut output = vec![0.0f32; output_size];
        let output_buffer_size_in_bytes = output_size * size_of::<f32>();

        let result = unsafe {
            embedder_embed(
                self.handle,
                flat_input_ids.as_ptr(),
                flat_attention_mask.as_ptr(),
                total_samples as c_long,
                seq_length as c_long,
                processing_batch_size as c_long,
                output.as_mut_ptr(),
                output_buffer_size_in_bytes as c_long,
            )
        };

        if result == 0 {
            Ok(output
                .chunks(self.embedding_dim)
                .map(|chunk| chunk.to_vec())
                .collect())
        } else {
            Err(E::msg(format!(
                "Batch embedding failed with error code: {}",
                result
            )))
        }
    }
}

impl Drop for EmbedderModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                embedder_free_model(self.handle);
            }
        }
    }
}

#[derive(Serialize)]
struct BenchmarkResult {
    model: String,
    device: String,
    max_seq_length: usize,
    batch_size: usize,
    total_time: f64,
    run_times: Vec<f64>,
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

fn write_embeddings(path: &Path, embeddings: &[Vec<f32>]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for embedding in embeddings {
        for val in embedding {
            if *val >= 0.0 {
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

    /// The size of each output vector produced by this model
    #[arg(short, long)]
    embedding_dim: usize,

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
        "output/{}/{}/embeddings-{}.txt",
        args.model, args.device, args.max_seq_length
    );
    let parent = std::path::Path::new(&output_path).parent().unwrap();
    fs::create_dir_all(parent).unwrap();

    PathBuf::from(output_path)
}

fn get_torchscript_model_path(args: &Args, repo_root: &Path) -> PathBuf {
    repo_root.join(format!("torchscript-models/output/{}/model.pt", args.model))
}

fn get_input_tokens_path(args: &Args, repo_root: &Path) -> PathBuf {
    repo_root.join(format!(
        "data/reference-output/{}/tokenized.txt",
        args.model
    ))
}

fn main() -> Result<()> {
    let repo_root = get_repo_root();

    let args = Args::parse();

    if args.num_runs < 1 {
        return Err(E::msg(format!("num runs must be at least 1, given {}", args.num_runs)));
    }

    let start_total = Instant::now();

    let output_path = get_output_path(&args);
    let input_tokens_path = get_input_tokens_path(&args, &repo_root);
    let model_path = get_torchscript_model_path(&args, &repo_root);

    eprintln!("Loading model");
    let model = EmbedderModel::load(&model_path, args.device, args.embedding_dim)?;
    eprintln!("Parsing input tokens");
    let input = TokenBatch::load(&input_tokens_path, args.max_seq_length)?;

    let mut run_times = Vec::new();
    let mut embeddings = None;

    for i in 0..args.num_runs {
        eprintln!("Run {}/{}", i + 1, args.num_runs);
        let start_run = Instant::now();
        embeddings = Some(model.embed_batch(&input.input_ids, &input.attention_mask, args.batch_size)?);
        let end_run = Instant::now();
        run_times.push(end_run.duration_since(start_run).as_secs_f64());
    }

    let end_total = Instant::now();
    let total_time = end_total.duration_since(start_total).as_secs_f64();

    let benchmark_result = BenchmarkResult {
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
