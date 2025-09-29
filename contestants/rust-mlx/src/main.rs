use anyhow::{Error as E, Result, anyhow};
use hf_hub::api::sync::Api as HfApi;
use memmap2::Mmap;
use mlx_rs::Array;
use mlx_rs::ops as mx;
use mlx_rs::ops::indexing::IndexOp;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

const USE_SDPA: bool = true;

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

fn mean_pooling(sequence_output: &Array, attention_mask: &Array) -> Result<Array> {
    let expanded_mask = attention_mask.expand_dims(-1)?;
    let masked_output = sequence_output * &expanded_mask;
    let sum_output = masked_output.sum_axis(1, false)?;
    let mask_sum = expanded_mask.sum_axis(1, false)?;
    Ok(sum_output / mask_sum)
}

fn normalize_embeddings(embeddings: &Array) -> Result<Array> {
    let norm = embeddings.square()?.sum_axis(1, true)?.sqrt()?;
    Ok(embeddings / &norm)
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
    word_embeddings: Array,
    position_embeddings: Array,
    token_type_embeddings: Array,
    layer_norm_weight: Array,
    layer_norm_bias: Array,
    layer_norm_eps: f64,
}

fn param(weights: &mut HashMap<String, Array>, k: &str) -> Result<Array> {
    weights.remove(k).ok_or(anyhow!("missing param key: {}", k))
}

impl BertEmbeddings {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>) -> Result<Self> {
        let word_embeddings = param(weights, "embeddings.word_embeddings.weight")?;
        let position_embeddings = param(weights, "embeddings.position_embeddings.weight")?;
        let token_type_embeddings = param(weights, "embeddings.token_type_embeddings.weight")?;
        let layer_norm_weight = param(weights, "embeddings.LayerNorm.weight")?;
        let layer_norm_bias = param(weights, "embeddings.LayerNorm.bias")?;
        let layer_norm_eps = config.layer_norm_eps;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_eps,
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
            Some(&self.layer_norm_weight),
            Some(&self.layer_norm_bias),
            self.layer_norm_eps as f32,
        )?;

        Ok(embeddings)
    }
}

struct BertSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query_weight: Array,
    query_bias: Array,
    key_weight: Array,
    key_bias: Array,
    value_weight: Array,
    value_bias: Array,
}

impl BertSelfAttention {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>, prefix: &str) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let query_weight = param(weights, &format!("{}attention.self.query.weight", prefix))?;
        let query_bias = param(weights, &format!("{}attention.self.query.bias", prefix))?;
        let key_weight = param(weights, &format!("{}attention.self.key.weight", prefix))?;
        let key_bias = param(weights, &format!("{}attention.self.key.bias", prefix))?;
        let value_weight = param(weights, &format!("{}attention.self.value.weight", prefix))?;
        let value_bias = param(weights, &format!("{}attention.self.value.bias", prefix))?;

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
        })
    }

    fn transpose_for_scores(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let new_shape = [
            shape[0],
            shape[1],
            self.num_attention_heads as i32,
            self.attention_head_size as i32,
        ];
        x.reshape(&new_shape)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(E::msg)
    }

    fn forward(&self, hidden_states: &Array, attention_mask: &Array) -> Result<Array> {
        let mixed_query_layer = mx::addmm(
            &self.query_bias,
            &hidden_states,
            &self.query_weight.t(),
            None,
            None,
        )?;
        let mixed_key_layer = mx::addmm(
            &self.key_bias,
            &hidden_states,
            &self.key_weight.t(),
            None,
            None,
        )?;
        let mixed_value_layer = mx::addmm(
            &self.value_bias,
            &hidden_states,
            &self.value_weight.t(),
            None,
            None,
        )?;

        let query_layer = self.transpose_for_scores(&mixed_query_layer)?;
        let key_layer = self.transpose_for_scores(&mixed_key_layer)?;

        let value_layer = self.transpose_for_scores(&mixed_value_layer)?;

        let context_layer = if USE_SDPA {
            mlx_rs::fast::scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                1.0 / (self.attention_head_size as f32).sqrt(),
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(
                    &attention_mask,
                )),
            )?
        } else {
            let mut attention_scores =
                query_layer.matmul(&key_layer.transpose_axes(&[0, 1, 3, 2])?)?;
            attention_scores = attention_scores / (self.attention_head_size as f32).sqrt();
            attention_scores = attention_scores + attention_mask;

            let attention_probs = mx::softmax_axis(&attention_scores, -1, None)?;
            attention_probs.matmul(&value_layer)?
        };

        let context_layer = context_layer.transpose_axes(&[0, 2, 1, 3])?;

        let shape = context_layer.shape();
        context_layer
            .reshape(&[shape[0], shape[1], self.all_head_size as i32])
            .map_err(E::msg)
    }
}

struct BertSelfOutput {
    dense_weight: Array,
    dense_bias: Array,
    layer_norm_weight: Array,
    layer_norm_bias: Array,
    layer_norm_eps: f32,
}

impl BertSelfOutput {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>, prefix: &str) -> Result<Self> {
        let dense_weight = param(weights, &format!("{}attention.output.dense.weight", prefix))?;
        let dense_bias = param(weights, &format!("{}attention.output.dense.bias", prefix))?;
        let layer_norm_weight = param(
            weights,
            &format!("{}attention.output.LayerNorm.weight", prefix),
        )?;
        let layer_norm_bias = param(
            weights,
            &format!("{}attention.output.LayerNorm.bias", prefix),
        )?;

        Ok(Self {
            dense_weight,
            dense_bias,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_eps: config.layer_norm_eps as f32,
        })
    }

    fn forward(&self, hidden_states: &Array, input_tensor: &Array) -> Result<Array> {
        let hidden_states = mx::addmm(
            &self.dense_bias,
            hidden_states,
            &self.dense_weight.t(),
            None,
            None,
        )?;
        let residual = &hidden_states + input_tensor;

        let output = mlx_rs::fast::layer_norm(
            residual,
            Some(&self.layer_norm_weight),
            Some(&self.layer_norm_bias),
            self.layer_norm_eps,
        )?;

        Ok(output)
    }
}

struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>, prefix: &str) -> Result<Self> {
        let self_attention = BertSelfAttention::new(config, weights, prefix)?;
        let output = BertSelfOutput::new(config, weights, prefix)?;

        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(&self, hidden_states: &Array, attention_mask: &Array) -> Result<Array> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_outputs, hidden_states)
    }
}

struct BertIntermediate {
    dense_weight: Array,
    dense_bias: Array,
}

impl BertIntermediate {
    fn new(
        _config: &ModelArgs,
        weights: &mut HashMap<String, Array>,
        prefix: &str,
    ) -> Result<Self> {
        let dense_weight = param(weights, &format!("{}intermediate.dense.weight", prefix))?;
        let dense_bias = param(weights, &format!("{}intermediate.dense.bias", prefix))?;

        Ok(Self {
            dense_weight,
            dense_bias,
        })
    }

    fn forward(&self, hidden_states: &Array) -> Result<Array> {
        let hidden_states = mx::addmm(
            &self.dense_bias,
            hidden_states,
            &self.dense_weight.t(),
            None,
            None,
        )?;
        mlx_rs::nn::gelu(&hidden_states).map_err(E::msg)
    }
}

struct BertOutput {
    dense_weight: Array,
    dense_bias: Array,
    layer_norm_weight: Array,
    layer_norm_bias: Array,
    layer_norm_eps: f64,
}

impl BertOutput {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>, prefix: &str) -> Result<Self> {
        let dense_weight = param(weights, &format!("{}output.dense.weight", prefix))?;
        let dense_bias = param(weights, &format!("{}output.dense.bias", prefix))?;
        let layer_norm_weight = param(weights, &format!("{}output.LayerNorm.weight", prefix))?;
        let layer_norm_bias = param(weights, &format!("{}output.LayerNorm.bias", prefix))?;

        Ok(Self {
            dense_weight,
            dense_bias,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_eps: config.layer_norm_eps,
        })
    }

    fn forward(&self, hidden_states: &Array, input_tensor: &Array) -> Result<Array> {
        let hidden_states = mx::addmm(
            &self.dense_bias,
            hidden_states,
            &self.dense_weight.t(),
            None,
            None,
        )?;
        let residual = &hidden_states + input_tensor;
        let output = mlx_rs::fast::layer_norm(
            residual,
            Some(&self.layer_norm_weight),
            Some(&self.layer_norm_bias),
            self.layer_norm_eps as f32,
        )?;

        Ok(output)
    }
}

struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>, prefix: &str) -> Result<Self> {
        let attention = BertAttention::new(config, weights, prefix)?;
        let intermediate = BertIntermediate::new(config, weights, prefix)?;
        let output = BertOutput::new(config, weights, prefix)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Array, attention_mask: &Array) -> Result<Array> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn new(config: &ModelArgs, weights: &mut HashMap<String, Array>) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|idx| BertLayer::new(config, weights, &format!("encoder.layer.{}.", idx)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers })
    }

    fn forward(&self, mut hidden_states: Array, attention_mask: &Array) -> Result<Array> {
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
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
        let encoder = BertEncoder::new(config, &mut clean_weights)?;
        Ok(Self {
            embeddings,
            encoder,
        })
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

    fn get_extended_attention_mask(attention_mask: &Array) -> Result<Array> {
        if attention_mask.ndim() != 2 {
            return Err(anyhow!(
                "attention mask must be 2d, got {}",
                attention_mask.ndim()
            ));
        }

        let extended_mask = attention_mask.expand_dims(1)?.expand_dims(1)?;
        assert!(extended_mask.ndim() == 4);
        let mask = (mx::ones_like(&extended_mask)? - &extended_mask) * -10_000.0;
        Ok(mask)
    }

    fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        println!("First layer embeddings \n{:?}", embedding_output);
        let extended_attention_mask = Self::get_extended_attention_mask(attention_mask)?;
        let encoder_outputs = self
            .encoder
            .forward(embedding_output, &extended_attention_mask)?;

        println!("Encoder outputs\n{:?}", encoder_outputs);
        let text_embeds = normalize_embeddings(&mean_pooling(&encoder_outputs, attention_mask)?)?;
        Ok(text_embeds)
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
        output.eval()?;
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
