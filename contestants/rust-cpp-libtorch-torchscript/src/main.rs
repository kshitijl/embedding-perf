use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_int, c_long};

unsafe extern "C" {
    fn embedder_load_model(model_path: *const c_char, use_mps: c_int) -> *mut std::ffi::c_void;
    fn embedder_free_model(model_handle: *mut std::ffi::c_void);
    fn embedder_embed(
        model_handle: *mut std::ffi::c_void,
        input_ids: *const i64,
        attention_mask: *const i64,
        batch_size: c_long,
        seq_length: c_long,
        output_embeddings: *mut f32,
        output_buffer_size_in_bytes: c_long,
    ) -> c_long;
}

struct EmbedderModel {
    handle: *mut std::ffi::c_void,
    embedding_dim: usize,
}

impl EmbedderModel {
    fn load(model_path: &str, device: Device, embedding_dim: usize) -> Result<Self> {
        let c_path = CString::new(model_path).map_err(E::msg)?;

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
    ) -> Result<Vec<Vec<f32>>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = input_ids.len();
        let seq_length = input_ids[0].len();

        for (ids, mask) in input_ids.iter().zip(attention_masks.iter()) {
            if ids.len() != seq_length || mask.len() != seq_length {
                return Err(E::msg("All sequences must have same length"));
            }
        }

        let flat_input_ids: Vec<i64> = input_ids.iter().flatten().copied().collect();
        let flat_attention_mask: Vec<i64> = attention_masks.iter().flatten().copied().collect();

        let output_size = batch_size * self.embedding_dim;
        let mut output = vec![0.0f32; output_size];
        let output_buffer_size_in_bytes = output_size * size_of::<f32>();

        let result = unsafe {
            embedder_embed(
                self.handle,
                flat_input_ids.as_ptr(),
                flat_attention_mask.as_ptr(),
                batch_size as c_long,
                seq_length as c_long,
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

#[derive(Debug)]
struct TokenBatch {
    input_ids: Vec<Vec<i64>>,
    attention_mask: Vec<Vec<i64>>,
}

impl TokenBatch {
    fn load(path: &str, max_length: usize) -> Result<Self> {
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

fn write_embeddings(path: &str, embeddings: &[Vec<f32>]) -> Result<()> {
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

#[derive(Parser, Debug)]
struct Args {
    /// Path to TorchScript model
    #[arg(long)]
    model_path: String,

    /// Path to jsonl file of tokenized input
    #[arg(short, long)]
    input: String,

    /// Truncate each tokenized input to this many tokens
    #[arg(short, long)]
    max_length: usize,

    /// Whether the model uses MPS. Traced TorchScript models are device-specific.
    #[arg(short, long)]
    device: Device,

    /// The size of each output vector produced by this model
    #[arg(short, long)]
    embedding_dim: usize,

    /// Output embeddings to this file
    #[arg(short, long)]
    output_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("Loading model");
    let model = EmbedderModel::load(&args.model_path, args.device, args.embedding_dim)?;
    eprintln!("Parsing input tokens");
    let input = TokenBatch::load(&args.input, args.max_length)?;

    eprintln!("Computing embeddings");
    let embeddings = model.embed_batch(&input.input_ids, &input.attention_mask)?;
    eprintln!("Writing embeddings to file");
    write_embeddings(&args.output_path, &embeddings)?;
    eprintln!("Done");
    Ok(())
}
