#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <memory>

#include "embedder_wrapper.h"

struct ModelWrapper {
    torch::jit::script::Module model;
    torch::Device device;

    ModelWrapper(torch::Device dev) : device(dev) {}
};

extern "C" {
void* embedder_load_model(const char* model_path, int use_mps) {
    torch::Device device_ = torch::kCPU;
    if (use_mps == 1) {
        if (!torch::mps::is_available()) {
            throw std::runtime_error("MPS requested but not available ");
        }
        device_ = torch::kMPS;
    } else if (use_mps == 0) {
        device_ = torch::kCPU;
    } else {
        throw std::runtime_error("use_mps should be 0 or 1, actually " + std::to_string(use_mps));
    }

    auto wrapper = std::make_unique<ModelWrapper>(device_);

    try {
        wrapper->model = torch::jit::load(model_path, device_);
        wrapper->model = torch::jit::optimize_for_inference(wrapper->model);
        wrapper->model.eval();
        std::cerr << "Model loaded successfully\n";

        return wrapper.release();
    } catch (const c10::Error& e) {
        std::cerr << "error loading model: " << e.what() << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return nullptr;
    }
}

void embedder_free_model(void* model_handle) {
    if (model_handle) {
        delete static_cast<ModelWrapper*>(model_handle);
        std::cerr << "Model freed." << std::endl;
    }
}

int embedder_embed(void* model_handle, const int64_t* input_ids, const int64_t* attention_mask,
                   int total_samples, int seq_length, int processing_batch_size,
                   float* output_embeddings, int64_t output_buffer_size_in_bytes) {
    if (!model_handle || !input_ids || !attention_mask || !output_embeddings) {
        std::cerr << "Null pointer passed to embed" << std::endl;
        return -1;
    }

    try {
        auto* wrapper = static_cast<ModelWrapper*>(model_handle);

        auto opts = torch::TensorOptions().dtype(torch::kLong);

        std::cerr << "Making full input tensors\n";
        torch::Tensor full_input_ids =
            torch::from_blob(const_cast<int64_t*>(input_ids), {total_samples, seq_length}, opts)
                .to(wrapper->device);

        torch::Tensor full_attention_mask = torch::from_blob(const_cast<int64_t*>(attention_mask),
                                                             {total_samples, seq_length}, opts)
                                                .to(wrapper->device);

        std::vector<torch::Tensor> output_batches;
        int embedding_dim = -1;

        torch::InferenceMode guard;
        std::cerr << "Num threads: " << torch::get_num_threads() << "\n";
        std::cerr << "Num interop threads: " << torch::get_num_interop_threads() << "\n";
        std::cerr << "MKL: " << torch::hasMKL() << "\n";
        std::cerr << "OpenMP: " << torch::hasOpenMP() << "\n";
        std::cerr << "XLA: " << torch::hasXLA() << "\n";

        std::cerr << "BLAS backend: " << torch::utils::get_blas_vendor() << std::endl;

        for (int start_idx = 0; start_idx < total_samples; start_idx += processing_batch_size) {
            int current_batch_size = std::min(processing_batch_size, total_samples - start_idx);

            std::cerr << "Processing batch " << start_idx / processing_batch_size + 1
                      << ", samples " << start_idx << "-" << start_idx + current_batch_size - 1
                      << "\n";

            torch::Tensor batch_input_ids =
                full_input_ids.slice(0, start_idx, start_idx + current_batch_size);
            torch::Tensor batch_attention_mask =
                full_attention_mask.slice(0, start_idx, start_idx + current_batch_size);

            c10::Dict<std::string, torch::Tensor> input_dict;
            input_dict.insert("input_ids", batch_input_ids);
            input_dict.insert("attention_mask", batch_attention_mask);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_dict);

            auto start = std::chrono::high_resolution_clock::now();
            auto output = wrapper->model.forward(inputs);
            auto end = std::chrono::high_resolution_clock::now();
            std::cerr << "Forward pass took: "
                      << std::chrono::duration<float, std::milli>(end - start).count() << " ms\n";

            // auto output = wrapper->model.forward(inputs);

            torch::Tensor embeddings;
            if (output.isGenericDict()) {
                auto output_dict = output.toGenericDict();
                embeddings = output_dict.at("sentence_embedding").toTensor();
            } else if (output.isTensor()) {
                embeddings = output.toTensor();
            } else if (output.isTuple()) {
                auto tuple_output = output.toTuple();
                embeddings = tuple_output->elements()[0].toTensor();
            } else {
                std::cerr << "Unexpected output type from model" << std::endl;
                return -3;
            }

            if (embedding_dim == -1) {
                embedding_dim = embeddings.size(1);
            }

            output_batches.push_back(embeddings.cpu());
        }

        std::cerr << "Concatenating output batches\n";
        torch::Tensor final_embeddings = torch::cat(output_batches, 0).cpu();

        std::cerr << "Copying output into provided buffer\n";
        size_t num_elements = final_embeddings.numel();
        const float* data_ptr = final_embeddings.data_ptr<float>();
        size_t num_bytes = num_elements * sizeof(float);
        if (num_bytes > output_buffer_size_in_bytes) {
            std::cerr << "Output buffer too small. Need " << num_bytes << " bytes, got "
                      << output_buffer_size_in_bytes << " bytes." << std::endl;
            return -6;
        }
        std::memcpy(output_embeddings, data_ptr, num_bytes);

        std::cerr << "Done\n";
        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error during embedding: " << e.what() << std::endl;
        return -4;
    } catch (const std::exception& e) {
        std::cerr << "Error during embedding: " << e.what() << std::endl;
        return -5;
    }
}
}
