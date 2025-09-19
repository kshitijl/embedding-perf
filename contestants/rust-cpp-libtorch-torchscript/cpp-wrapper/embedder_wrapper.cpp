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
                   int batch_size, int seq_length, float* output_embeddings,
                   int64_t output_buffer_size_in_bytes) {
    if (!model_handle || !input_ids || !attention_mask || !output_embeddings) {
        std::cerr << "Null pointer passed to embed" << std::endl;
        return -1;
    }

    try {
        auto* wrapper = static_cast<ModelWrapper*>(model_handle);

        auto opts = torch::TensorOptions().dtype(torch::kLong);

        std::cerr << "Making input tensors\n";
        torch::Tensor input_ids_tensor =
            torch::from_blob(const_cast<int64_t*>(input_ids), {batch_size, seq_length}, opts)
                .to(wrapper->device);

        torch::Tensor attention_mask_tensor =
            torch::from_blob(const_cast<int64_t*>(attention_mask), {batch_size, seq_length}, opts)
                .to(wrapper->device);

        std::cerr << "Making input dictionary\n";
        c10::Dict<std::string, torch::Tensor> input_dict;
        input_dict.insert("input_ids", input_ids_tensor);
        input_dict.insert("attention_mask", attention_mask_tensor);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_dict);

        torch::NoGradGuard no_grad;
        std::cerr << "Calling the model\n";
        auto output = wrapper->model.forward(inputs);

        torch::Tensor embeddings;

        std::cerr << "Getting the output out\n";
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

        std::cerr << "Copying output into provided buffer\n";
        embeddings = embeddings.cpu().contiguous();

        size_t num_elements = embeddings.numel();
        const float* data_ptr = embeddings.data_ptr<float>();
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
