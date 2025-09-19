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
int embedder_add_numbers(int a, int b) { return a + b; }

void* embedder_load_model(const char* model_path, int use_mps) {
    torch::Device device_ = torch::kCPU;
    if (use_mps == 1) {
        if (!torch::mps::is_available()) {
            throw std::runtime_error("MPS requested but not available.");
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

        std::cerr << "Done";
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

// #include <c10/core/DeviceType.h>
// #include <torch/csrc/jit/api/module.h>
// #include <torch/mps.h>
// #include <torch/script.h>
// #include <torch/torch.h>

// #include <iomanip>
// #include <ios>
// #include <iostream>

// struct TokenizedInput {
//     torch::Tensor input_ids;
//     torch::Tensor attention_mask;
//     torch::Tensor token_type_ids;
// };

// class SentenceEmbedder {
//    private:
//     torch::jit::script::Module model_;
//     torch::Device device_;

//    public:
//     SentenceEmbedder(const std::string& model_path, bool use_mps = false) : device_(torch::kCPU)
//     {
//         if (use_mps) {
//             if (!torch::mps::is_available()) {
//                 throw "MPS requested but not available.";
//             }
//             device_ = torch::kMPS;
//         } else {
//             device_ = torch::kCPU;
//         }

//         try {
//             model_ = torch::jit::load(model_path, torch::kMPS);
//             model_.eval();
//         } catch (const c10::Error& e) {
//             std::cerr << "error loading model: " << e.what() << "\n";
//             throw "could not load model";
//         }
//     }

//     torch::Tensor embed(const std::vector<TokenizedInput>& tokenized_inputs) {
//         try {
//             std::vector<torch::Tensor> input_ids_list, attention_mask_list, token_type_ids_list;

//             for (const auto& input : tokenized_inputs) {
//                 input_ids_list.push_back(input.input_ids);
//                 attention_mask_list.push_back(input.attention_mask);
//                 token_type_ids_list.push_back(input.token_type_ids);
//             }

//             auto batch_input_ids = torch::stack(input_ids_list).to(device_);
//             auto batch_attention_mask = torch::stack(attention_mask_list).to(device_);
//             // auto batch_token_type_ids = torch::stack(token_type_ids_list).to(device_);

//             c10::Dict<std::string, torch::Tensor> input_dict;
//             input_dict.insert("input_ids", batch_input_ids);
//             input_dict.insert("attention_mask", batch_attention_mask);
//             // input_dict.insert("token_type_ids", batch_token_type_ids);

//             std::vector<torch::jit::IValue> inputs;
//             inputs.push_back(input_dict);
//             torch::jit::IValue output = model_.forward(inputs);

//             torch::Tensor embeddings;
//             if (output.isGenericDict()) {
//                 auto output_dict = output.toGenericDict();
//                 auto embedding_ivalue = output_dict.at("sentence_embedding");
//                 embeddings = embedding_ivalue.toTensor();
//             } else if (output.isTensor()) {
//                 embeddings = output.toTensor();
//             } else if (output.isTuple()) {
//                 auto tuple_output = output.toTuple();
//                 embeddings = tuple_output->elements()[0].toTensor();
//             } else {
//                 std::cout << output << std::endl;
//                 throw std::runtime_error("Unexpected output type from model.");
//             }

//             return embeddings.to(torch::kCPU);
//         } catch (const std::exception& e) {
//             std::cerr << "Error during inference: " << e.what() << std::endl;
//             throw;
//         }
//     }
// };

// void writeEmbeddings(const torch::Tensor& embeddings, const std::string& output_path) {
//     std::ofstream file(output_path);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open output file: " + output_path);
//     }
//     file << std::scientific << std::setprecision(8);

//     auto cpu_embeddings = embeddings.cpu().contiguous();

//     auto sizes = cpu_embeddings.sizes();
//     if (sizes.size() != 2) {
//         throw std::runtime_error("Expected 2D tensor, got " + std::to_string(sizes.size()) +
//         "D");
//     }

//     const long long num_sentences = sizes[0];
//     const long long embedding_dim = sizes[1];

//     const float* data_ptr = cpu_embeddings.data_ptr<float>();

//     for (int i = 0; i < num_sentences; ++i) {
//         for (int j = 0; j < embedding_dim; ++j) {
//             float number = data_ptr[i * embedding_dim + j];

//             if (number >= 0.0) {
//                 // Two spaces before positive numbers, one space before negative makes them line
//                 up
//                 // nicely.
//                 file << " ";
//             }
//             file << number;

//             if (j < embedding_dim - 1) {
//                 file << " ";
//             }
//         }
//         file << "\n";
//     }

//     std::cout << "Embeddings written to: " << output_path << std::endl;
//     std::cout << "Shape: [" << num_sentences << ", " << embedding_dim << "]" << std::endl;
// }

// // int main(int argc, char** argv) {
// //     if (argc != 5) {
// //         std::cerr << "usage: main <path-to-torchscript> <tokens> <output> <device>\n";
// //         return 1;
// //     }
// //     const std::string model_path = argv[1];
// //     const std::string token_path = argv[2];
// //     const std::string output_path = argv[3];
// //     const std::string device = argv[4];

// //     bool use_mps = false;
// //     if (device == "cpu") {
// //         use_mps = false;
// //     } else if (device == "mps") {
// //         use_mps = true;
// //     } else {
// //         std::cerr << "Device must be one of cpu, mps";
// //         return 1;
// //     }

// //     // auto tokenized_inputs = readTokenizedInputs(token_path);
// //     // std::cout << " input_ids shape: " << tokenized_inputs[0].input_ids.sizes()
// //     //           << ", attention mask shape: " << tokenized_inputs[0].attention_mask.sizes()
// <<
// //     //           "\n";

// //     // SentenceEmbedder embedder(model_path, use_mps);

// //     // std::cout << "Generating embeddings... \n";
// //     // auto embeddings = embedder.embed(tokenized_inputs);

// //     // writeEmbeddings(embeddings, output_path);

// //     std::cout << "ok\n";
// //     return 0;
// // }
