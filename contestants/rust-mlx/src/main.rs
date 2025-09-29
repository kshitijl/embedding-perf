use anyhow::{Error as E, Result, anyhow};
use hf_hub::api::sync::Api as HfApi;
use memmap2::Mmap;
use mlx_rs::module::{Module, Param};
use mlx_rs::ops as mx;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelArgs {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    hidden_dropout_prob: f32,
    attention_probs_dropout_prob: f32,
    type_vocab_size: usize,
    layer_norm_eps: f64,
    // There are a bunch of additional fields in config files that some
    // implementations read, but we don't need them so we'll ignore them. Since
    // we don't write configs, only read them, it doesn't matter.
}

fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Array>> {
    let file = File::open(path)?;
    let buffer = unsafe { Mmap::map(&file)? };
    let safetensors = SafeTensors::deserialize(&buffer)?;
    let mut weights = HashMap::new();

    for name in safetensors.names() {
        let tensor = safetensors.tensor(&name)?;
        let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();

        let array = match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = tensor
                    .data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Array::from_slice(&data, &shape)
            }
            safetensors::Dtype::I64 => {
                let data: Vec<i64> = tensor
                    .data()
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                Array::from_slice(&data, &shape)
            }
            _ => {
                panic!("unexpected type {:}", tensor.dtype())
            }
        };

        println!(
            "loaded {} tensor {}, shape {:?}",
            tensor.dtype(),
            name,
            tensor.shape()
        );
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

struct BertTokenizer {
    tokenizer: Tokenizer,
}

impl BertTokenizer {
    fn new(model_name: &str) -> Result<Self> {
        let api = HfApi::new()?;
        let repo = api.model(model_name.to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        tokenizer.get_padding_mut().unwrap().strategy = tokenizers::PaddingStrategy::BatchLongest;

        println!(
            "loaded tokenizer {:?} {:?}",
            tokenizer.get_truncation(),
            tokenizer.get_padding()
        );

        Ok(Self { tokenizer })
    }

    fn encode_batch(&self, texts: Vec<&str>) -> Result<Vec<Encoding>> {
        Ok(self.tokenizer.encode_batch(texts, true).map_err(E::msg)?)
    }

    fn to_arrays(&self, encodings: &[Encoding]) -> Result<(Array, Array)> {
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let batch_size = encodings.len();

        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let start = i * max_len;

            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                if j < max_len {
                    input_ids[start + j] = id;
                    attention_mask[start + j] = m;
                }
            }
        }

        let batch_size = batch_size as i32;
        let max_len = max_len as i32;
        let input_ids = Array::from_slice(&input_ids, &[batch_size, max_len]);
        let attention_mask = Array::from_slice(&attention_mask, &[batch_size, max_len]);

        Ok((input_ids, attention_mask))
    }
}

struct BertEmbeddings {
    word_embeddings: Param<Array>,
    position_embeddings: Param<Array>,
    token_type_embeddings: Param<Array>,
    layer_norm_weight: Param<Array>,
    layer_norm_bias: Param<Array>,
    layer_norm_eps: f64,
    dropout_prob: f32,
}

fn take_param(weights: &mut HashMap<String, Array>, k: &str) -> Result<Array> {
    weights.remove(k).ok_or(anyhow!("missing param key: {}", k))
}

impl BertEmbeddings {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>) -> Result<Self> {
        let word_embeddings = Param::new(take_param(weights, "embeddings.word_embeddings.weight")?);

        let position_embeddings = Param::new(take_param(
            weights,
            "embeddings.position_embeddings.weight",
        )?);

        let token_type_embeddings = Param::new(take_param(
            weights,
            "embeddings.token_type_embeddings.weight",
        )?);

        let layer_norm_weight = Param::new(take_param(weights, "embeddings.LayerNorm.weight")?);

        let layer_norm_bias = Param::new(take_param(weights, "embeddings.LayerNorm.bias")?);

        let layer_norm_eps = config.layer_norm_eps;
        let dropout_prob = config.hidden_dropout_prob;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_eps,
            dropout_prob,
        })
    }

    fn forward(&self, input_ids: &Array) -> Result<Array> {
        let seq_len = input_ids.shape()[1];

        let position_ids = mx::arange::<_, i32>(0, seq_len as i32, 1)?;
        let token_type_ids = mx::zeros_like(input_ids)?;

        let words_embedding = self.word_embeddings.index(input_ids);
        let position_embeddings = self.position_embeddings.index(position_ids);
        let token_type_embeddings = self.token_type_embeddings.index(&token_type_ids);

        let embeddings = &words_embedding + &position_embeddings + &token_type_embeddings;
        let embeddings = mlx_rs::fast::layer_norm(
            &embeddings,
            Some(&self.layer_norm_weight.value),
            Some(&self.layer_norm_bias.value),
            self.layer_norm_eps as f32,
        )?;

        Ok(embeddings)
    }
}

struct BertModel {
    embeddings: BertEmbeddings,
    // encoder: BertEncoder,
    // pooler: BertPooler,
}

impl BertModel {
    fn new(config: &ModelArgs, weights: HashMap<String, Array>) -> Result<Self> {
        let mut clean_weights = HashMap::new();
        for (name, weight) in weights {
            let clean_name = if name.starts_with("0.") {
                name[2..].to_string()
            } else {
                name
            };
            clean_weights.insert(clean_name, weight);
        }

        let embeddings = BertEmbeddings::new(config, &mut clean_weights)?;

        // let encoder = BertEncoder
        Ok(Self { embeddings })
    }

    fn load(model_name: &str) -> Result<Self> {
        println!("loading {}", model_name);
        let api = HfApi::new()?;
        let repo = api.model(model_name.to_string());

        let config_path = repo.get("config.json")?;
        let config: ModelArgs = serde_json::from_str(&fs::read_to_string(config_path)?)?;
        println!("config {:?}", &config);

        let weights_path = repo.get("model.safetensors")?;
        let weights = load_safetensors(weights_path)?;

        let model = Self::new(&config, weights)?;

        Ok(model)
    }

    fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let embedding_output = self.embeddings.forward(input_ids)?;

        Ok(embedding_output)
    }
}

struct BertEmbedder {
    model: BertModel,
    tokenizer: BertTokenizer,
}

impl BertEmbedder {
    fn load(model_name: &str) -> Result<Self> {
        let tokenizer = BertTokenizer::new(model_name)?;
        let model = BertModel::load(model_name)?;

        Ok(Self { model, tokenizer })
    }

    fn embed(&self, sentences: Vec<&str>) -> Result<Array> {
        let encodings = self.tokenizer.encode_batch(sentences)?;
        let (input_ids, attention_mask) = self.tokenizer.to_arrays(&encodings)?;
        println!("input_ids {:?} \n{:?}\n", input_ids.shape(), input_ids);
        println!(
            "attention_mask {:?} \n{:?}\n",
            attention_mask.shape(),
            attention_mask
        );
        let output = self.model.forward(&input_ids, &attention_mask)?;
        Ok(output)
    }
}

fn main() -> Result<()> {
    let model = BertEmbedder::load("sentence-transformers/all-MiniLM-L6-v2").unwrap();
    let embeddings = model
        .embed(vec![
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
        ])
        .unwrap();

    println!("embeddings \n{:?}", embeddings);
    Ok(())
}
