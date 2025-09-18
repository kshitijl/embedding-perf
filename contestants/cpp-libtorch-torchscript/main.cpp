#include <c10/core/DeviceType.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/mps.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iomanip>
#include <ios>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct TokenizedInput {
    torch::Tensor input_ids;
    torch::Tensor attention_mask;
    torch::Tensor token_type_ids;
};

class SentenceEmbedder {
   private:
    torch::jit::script::Module model_;
    torch::Device device_;

   public:
    SentenceEmbedder(const std::string& model_path, bool use_mps = false) : device_(torch::kCPU) {
        if (use_mps) {
            if (!torch::mps::is_available()) {
                throw "MPS requested but not available.";
            }
            device_ = torch::kMPS;
        } else {
            device_ = torch::kCPU;
        }

        try {
            model_ = torch::jit::load(model_path, torch::kMPS);
            model_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "error loading model: " << e.what() << "\n";
            throw "could not load model";
        }
    }

    torch::Tensor embed(const std::vector<TokenizedInput>& tokenized_inputs) {
        try {
            std::vector<torch::Tensor> input_ids_list, attention_mask_list, token_type_ids_list;

            for (const auto& input : tokenized_inputs) {
                input_ids_list.push_back(input.input_ids);
                attention_mask_list.push_back(input.attention_mask);
                token_type_ids_list.push_back(input.token_type_ids);
            }

            auto batch_input_ids = torch::stack(input_ids_list).to(device_);
            auto batch_attention_mask = torch::stack(attention_mask_list).to(device_);
            auto batch_token_type_ids = torch::stack(token_type_ids_list).to(device_);

            c10::Dict<std::string, torch::Tensor> input_dict;
            input_dict.insert("input_ids", batch_input_ids);
            input_dict.insert("attention_mask", batch_attention_mask);
            input_dict.insert("token_type_ids", batch_token_type_ids);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_dict);
            torch::jit::IValue output = model_.forward(inputs);

            torch::Tensor embeddings;
            if (output.isGenericDict()) {
                auto output_dict = output.toGenericDict();
                auto embedding_ivalue = output_dict.at("sentence_embedding");
                embeddings = embedding_ivalue.toTensor();
            } else if (output.isTensor()) {
                embeddings = output.toTensor();
            } else if (output.isTuple()) {
                auto tuple_output = output.toTuple();
                embeddings = tuple_output->elements()[0].toTensor();
            } else {
                std::cout << output << std::endl;
                throw std::runtime_error("Unexpected output type from model.");
            }

            return embeddings.to(torch::kCPU);
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            throw;
        }
    }
};

std::vector<TokenizedInput> readTokenizedInputs(const std::string& filename) {
    std::vector<TokenizedInput> answer;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        try {
            json j = json::parse(line);

            auto input_ids_nested = j["input_ids"].get<std::vector<std::vector<int64_t>>>();
            auto input_ids_vec = input_ids_nested[0];

            auto attention_mask_nested =
                j["attention_mask"].get<std::vector<std::vector<int64_t>>>();
            auto attention_mask_vec = attention_mask_nested[0];

            std::vector<int64_t> token_type_ids_vec;
            if (j.contains("token_type_ids")) {
                auto token_type_ids_nested =
                    j["token_type_ids"].get<std::vector<std::vector<int64_t>>>();
                token_type_ids_vec = token_type_ids_nested[0];
            }

            auto input_ids = torch::tensor(input_ids_vec, torch::kLong);
            auto attention_mask = torch::tensor(attention_mask_vec, torch::kLong);
            auto token_type_ids = token_type_ids_vec.empty()
                                      ? torch::zeros_like(input_ids)
                                      : torch::tensor(token_type_ids_vec, torch::kLong);

            answer.push_back({input_ids, attention_mask, token_type_ids});
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return answer;
}

void writeEmbeddings(const torch::Tensor& embeddings, const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_path);
    }
    file << std::scientific << std::setprecision(8);

    auto cpu_embeddings = embeddings.cpu().contiguous();

    auto sizes = cpu_embeddings.sizes();
    if (sizes.size() != 2) {
        throw std::runtime_error("Expected 2D tensor, got " + std::to_string(sizes.size()) + "D");
    }

    const long long num_sentences = sizes[0];
    const long long embedding_dim = sizes[1];

    const float* data_ptr = cpu_embeddings.data_ptr<float>();

    for (int i = 0; i < num_sentences; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            float number = data_ptr[i * embedding_dim + j];

            if (number >= 0.0) {
                // Two spaces before positive numbers, one space before negative makes them line up
                // nicely.
                file << " ";
            }
            file << number;

            if (j < embedding_dim - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    std::cout << "Embeddings written to: " << output_path << std::endl;
    std::cout << "Shape: [" << num_sentences << ", " << embedding_dim << "]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: main <path-to-torchscript> <tokens> <output> <device>\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string token_path = argv[2];
    const std::string output_path = argv[3];
    const std::string device = argv[4];

    bool use_mps = false;
    if (device == "cpu") {
        use_mps = false;
    } else if (device == "mps") {
        use_mps = true;
    } else {
        std::cerr << "Device must be one of cpu, mps";
        return 1;
    }

    auto tokenized_inputs = readTokenizedInputs(token_path);
    std::cout << " input_ids shape: " << tokenized_inputs[0].input_ids.sizes()
              << ", attention mask shape: " << tokenized_inputs[0].attention_mask.sizes() << "\n";

    SentenceEmbedder embedder(model_path, use_mps);

    std::cout << "Generating embeddings... \n";
    auto embeddings = embedder.embed(tokenized_inputs);

    writeEmbeddings(embeddings, output_path);

    std::cout << "ok\n";
    return 0;
}
