use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_uint};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tch::{self, IValue, Tensor};

struct EmbedderModel {
    model: tch::CModule,
    device: tch::Device,
    embedding_dim: usize,
}

impl EmbedderModel {
    fn load(model_path: &Path, device: Device, embedding_dim: usize) -> Result<Self> {
        let tch_device = match device {
            Device::Cpu => tch::Device::Cpu,
            Device::Mps => tch::Device::Mps,
        };

        let mut model = tch::CModule::load_on_device(&model_path, tch_device)?;
        model.set_eval();

        Ok(EmbedderModel {
            model,
            embedding_dim,
            device: tch_device,
        })
    }

    fn embed_batch(
        &self,
        input_ids: &[Vec<i64>],
        attention_masks: &[Vec<i64>],
        processing_batch_size: usize,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
        }
        let _guard = tch::no_grad_guard();

        let total_samples = input_ids.len();
        let seq_length = input_ids[0].len();

        for (ids, mask) in input_ids.iter().zip(attention_masks.iter()) {
            if ids.len() != seq_length || mask.len() != seq_length {
                return Err(E::msg("All sequences must have same length"));
            }
        }

        let flat_input_ids: Vec<i64> = input_ids.iter().flatten().copied().collect();
        let flat_attention_mask: Vec<i64> = attention_masks.iter().flatten().copied().collect();

        let input_ids_tensor = Tensor::from_slice(&flat_input_ids)
            .reshape(&[input_ids.len() as i64, -1])
            .to(self.device);
        let attention_masks_tensor = Tensor::from_slice(&flat_attention_mask)
            .reshape(&[input_ids.len() as i64, -1])
            .to(self.device);

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

            let batch_input_ids =
                input_ids_tensor.slice(0, Some(batch_start as i64), Some(batch_end as i64), 1);
            let batch_attention_masks = attention_masks_tensor.slice(
                0,
                Some(batch_start as i64),
                Some(batch_end as i64),
                1,
            );
            let input_dict = IValue::GenericDict(vec![
                (
                    IValue::String("input_ids".to_string()),
                    IValue::Tensor(batch_input_ids),
                ),
                (
                    IValue::String("attention_mask".to_string()),
                    IValue::Tensor(batch_attention_masks),
                ),
            ]);

            let result = self.model.forward_is(&[input_dict])?;
            let mut embeddings = None;
            match result {
                IValue::GenericDict(items) => {
                    for (k, v) in items {
                        if k == IValue::String("sentence_embedding".to_string()) {
                            match v {
                                IValue::Tensor(t) => embeddings = Some(t),
                                _ => {
                                    panic!("unexpected")
                                }
                            };
                            break;
                        }
                    }
                }
                _ => {
                    panic!("unexpected output type")
                }
            };
            let batch_embeddings = embeddings.unwrap();
            all_batch_embeddings.push(batch_embeddings);
        }

        let embeddings = Tensor::cat(&all_batch_embeddings, 0);

        let mut answer = vec![0f32; embeddings.numel()];

        embeddings.copy_data(&mut answer, embeddings.numel());
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

fn write_embeddings(path: &Path, embeddings: &[f32], embedding_dim: usize) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    assert!(embeddings.len() % embedding_dim == 0);

    let nvecs = embeddings.len() / embedding_dim;
    for vec_idx in 0..nvecs {
        let embedding = &embeddings[vec_idx * embedding_dim..(vec_idx + 1) * embedding_dim];
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
        "output/{}/{}/embeddings-{}-{}.txt",
        args.model, args.device, args.max_seq_length, args.batch_size
    );
    let parent = std::path::Path::new(&output_path).parent().unwrap();
    fs::create_dir_all(parent).unwrap();

    PathBuf::from(output_path)
}

fn get_torchscript_model_path(args: &Args, repo_root: &Path) -> PathBuf {
    repo_root.join(format!("models/torchscript/output/{}/model.pt", args.model))
}

fn get_input_tokens_path(args: &Args, repo_root: &Path) -> PathBuf {
    repo_root.join(format!(
        "data/reference-output/{}/tokenized.txt",
        args.model
    ))
}

unsafe extern "C" {
    fn _dyld_image_count() -> c_uint;
    fn _dyld_get_image_name(image_index: c_uint) -> *const c_char;
}

#[cfg(target_os = "macos")]
fn find_libtorch_paths() -> Vec<String> {
    let count = unsafe { _dyld_image_count() };
    let mut answer = Vec::new();

    for i in 0..count {
        let name_ptr = unsafe { _dyld_get_image_name(i) };
        if !name_ptr.is_null() {
            if let Ok(name) = unsafe { CStr::from_ptr(name_ptr).to_str() } {
                if name.contains("libtorch") && name.ends_with(".dylib") {
                    answer.push(name.to_string());
                }
            }
        }
    }

    answer
}

enum BlasBackend {
    Accelerate,
    Mkl,
    OpenBlas,
}

impl fmt::Display for BlasBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match *self {
            BlasBackend::Accelerate => write!(f, "accelerate"),
            BlasBackend::Mkl => write!(f, "mkl"),
            BlasBackend::OpenBlas => write!(f, "openblas"),
        }
    }
}

fn detect_lobtorch_blas() -> Option<BlasBackend> {
    let libtorch_paths = find_libtorch_paths();

    for path in libtorch_paths {
        let otool_output = Command::new("otool").args(&["-L", &path]).output().ok()?;
        let otool_output = String::from_utf8_lossy(&otool_output.stdout).to_lowercase();

        if otool_output.contains("accelerate.framework") {
            return Some(BlasBackend::Accelerate);
        }

        if otool_output.contains("openblas") {
            return Some(BlasBackend::OpenBlas);
        }

        if otool_output.contains("libmkl") {
            return Some(BlasBackend::Mkl);
        }
    }

    None
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.num_runs < 1 {
        return Err(E::msg(format!(
            "num runs must be at least 1, given {}",
            args.num_runs
        )));
    }

    let blas = detect_lobtorch_blas()
        .map(|x| format!("{}", x))
        .unwrap_or("unknown".to_string());
    eprintln!("Detected BLAS: {}", blas);

    let repo_root = get_repo_root();
    let output_path = get_output_path(&args);
    let input_tokens_path = get_input_tokens_path(&args, &repo_root);
    let model_path = get_torchscript_model_path(&args, &repo_root);

    let start_total = Instant::now();
    eprintln!("Loading model");
    let model = EmbedderModel::load(&model_path, args.device, args.embedding_dim)?;
    eprintln!("Parsing input tokens");
    let input = TokenBatch::load(&input_tokens_path, args.max_seq_length)?;

    let mut run_times = Vec::new();
    let mut embeddings = None;

    for i in 0..args.num_runs {
        eprintln!("Run {}/{}", i + 1, args.num_runs);
        let start_run = Instant::now();
        embeddings =
            Some(model.embed_batch(&input.input_ids, &input.attention_mask, args.batch_size)?);
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
        contestant: format!("tch-ts-{}", blas),
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
    write_embeddings(&output_path, &embeddings.unwrap(), model.embedding_dim)?;
    eprintln!("Done");
    Ok(())
}
