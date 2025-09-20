#pragma once
#include <cstdint>
extern "C" {
void* embedder_load_model(const char* model_path, int use_mps);
void embedder_free_model(void* model_handle);
int embedder_embed(void* model_handle, const int64_t* input_ids, const int64_t* attention_mask,
                   int total_samples, int seq_length, int processing_batch_size, float* output_embeddings,
                   int64_t output_buffer_size_in_bytes);
}
