#include <c10/core/DeviceType.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/mps.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

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
            model_ = torch::jit::load(model_path, device_);
            model_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "error loading model: " << e.what() << "\n";
            throw "could not load model";
        }
    }

    torch::Tensor embed_batch(const std::vector<TokenizedInput>& tokenized_inputs,
                              size_t batch_size) {
        try {
            // Create full tensors from all inputs (stack once)
            std::vector<torch::Tensor> input_ids_list, attention_mask_list, token_type_ids_list;
            for (const auto& input : tokenized_inputs) {
                input_ids_list.push_back(input.input_ids);
                attention_mask_list.push_back(input.attention_mask);
                token_type_ids_list.push_back(input.token_type_ids);
            }

            auto full_input_ids = torch::stack(input_ids_list).to(device_);
            auto full_attention_mask = torch::stack(attention_mask_list).to(device_);
            // auto full_token_type_ids = torch::stack(token_type_ids_list).to(device_);

            std::vector<torch::Tensor> output_batches;

            torch::InferenceMode guard;

            // Process in batches using slices (no copying)
            for (size_t start_idx = 0; start_idx < tokenized_inputs.size();
                 start_idx += batch_size) {
                size_t current_batch_size =
                    std::min(batch_size, tokenized_inputs.size() - start_idx);

                // Create slices (views) into the full tensors - no data copying
                torch::Tensor batch_input_ids =
                    full_input_ids.slice(0, start_idx, start_idx + current_batch_size);
                torch::Tensor batch_attention_mask =
                    full_attention_mask.slice(0, start_idx, start_idx + current_batch_size);
                // torch::Tensor batch_token_type_ids =
                //     full_token_type_ids.slice(0, start_idx, start_idx + current_batch_size);

                c10::Dict<std::string, torch::Tensor> input_dict;
                input_dict.insert("input_ids", batch_input_ids);
                input_dict.insert("attention_mask", batch_attention_mask);
                // input_dict.insert("token_type_ids", batch_token_type_ids);

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

                output_batches.push_back(embeddings.cpu());
            }

            return torch::cat(output_batches, 0);
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            throw;
        }
    }
};

std::vector<TokenizedInput> readTokenizedInputs(const std::string& filename,
                                                unsigned long max_length) {
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

            // TODO this padding might not be correct for all sentence transformer models
            if (input_ids_vec.size() > max_length) {
                // Truncate to leave room for SEP token
                input_ids_vec.resize(max_length - 1);
                attention_mask_vec.resize(max_length - 1);

                // Add SEP token (102) since we truncated
                input_ids_vec.push_back(102);
                attention_mask_vec.push_back(1);
            }

            // Pad to max_length (works whether we truncated or not)
            input_ids_vec.resize(max_length, 0);
            attention_mask_vec.resize(max_length, 0);

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

std::string getRepoRoot() {
    std::string result;

    // Execute git command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("git rev-parse --show-toplevel", "r"),
                                                  pclose);

    if (!pipe) {
        throw std::runtime_error("Failed to run git command");
    }

    // Read output
    char buffer[128];
    while (fgets(buffer, sizeof buffer, pipe.get()) != nullptr) {
        result += buffer;
    }

    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    if (result.empty()) {
        throw std::runtime_error("Git command returned empty result");
    }

    return result;
}

std::string getOutputPath(const std::string& model, const std::string& device,
                          unsigned long max_seq_length, unsigned long batch_size) {
    std::string path = "output/" + model + "/" + device + "/embeddings-" +
                       std::to_string(max_seq_length) + "-" + std::to_string(batch_size) + ".txt";

    // Create directories if they don't exist
    std::filesystem::path fs_path(path);
    std::filesystem::create_directories(fs_path.parent_path());

    return path;
}

std::string getTorchscriptModelPath(const std::string& model, const std::string& repo_root) {
    return repo_root + "/models/torchscript/output/" + model + "/model.pt";
}

std::string getInputTokensPath(const std::string& model, const std::string& repo_root) {
    return repo_root + "/data/reference-output/" + model + "/tokenized.txt";
}

enum class BlasBackend { Accelerate, Mkl, OpenBlas, Unknown };

std::string blasBackendToString(BlasBackend backend) {
    switch (backend) {
        case BlasBackend::Accelerate:
            return "accelerate";
        case BlasBackend::Mkl:
            return "mkl";
        case BlasBackend::OpenBlas:
            return "openblas";
        case BlasBackend::Unknown:
            return "unknown";
    }
    return "unknown";
}

#ifdef __APPLE__
std::vector<std::string> findLibtorchPaths() {
    std::vector<std::string> paths;
    uint32_t count = _dyld_image_count();

    for (uint32_t i = 0; i < count; ++i) {
        const char* name = _dyld_get_image_name(i);
        if (name) {
            std::string path(name);
            if (path.find("libtorch") != std::string::npos &&
                path.find(".dylib") != std::string::npos) {
                paths.push_back(path);
            }
        }
    }

    return paths;
}

BlasBackend detectLibtorchBlas() {
    auto libtorch_paths = findLibtorchPaths();

    for (const auto& path : libtorch_paths) {
        std::string command = "otool -L \"" + path + "\"";

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) continue;

        std::string result;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);

        // Convert to lowercase for easier matching
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);

        if (result.find("accelerate.framework") != std::string::npos) {
            return BlasBackend::Accelerate;
        }

        if (result.find("openblas") != std::string::npos) {
            return BlasBackend::OpenBlas;
        }

        if (result.find("libmkl") != std::string::npos) {
            return BlasBackend::Mkl;
        }
    }

    return BlasBackend::Unknown;
}
#else
BlasBackend detectLibtorchBlas() {
    // On non-macOS systems, we could add Linux/Windows detection here
    // For now, return unknown
    return BlasBackend::Unknown;
}
#endif

int main(int argc, char** argv) {
    CLI::App app{"Sentence Transformer Embedder"};

    std::string model;
    std::string device;
    unsigned long max_seq_length;
    int embedding_dim;
    unsigned long batch_size;
    unsigned long num_runs = 1;

    app.add_option("--model", model, "Model name")->required();
    app.add_option("--device", device, "Device (cpu or mps)")->required();
    app.add_option("--max-seq-length", max_seq_length, "Max sequence length")->required();
    app.add_option("--embedding-dim", embedding_dim, "Embedding dimension")->required();
    app.add_option("--batch-size", batch_size, "Batch size for processing")->required();
    app.add_option("--num-runs", num_runs, "Number of runs for benchmarking")->default_val(1);

    CLI11_PARSE(app, argc, argv);

    // Validate device
    if (device != "cpu" && device != "mps") {
        std::cerr << "Device must be 'cpu' or 'mps'" << std::endl;
        return 1;
    }

    if (num_runs < 1) {
        std::cerr << "num runs must be at least 1, given " << num_runs << std::endl;
        return 1;
    }

    bool use_mps = (device == "mps");

    // Detect BLAS backend
    BlasBackend blas = detectLibtorchBlas();
    std::string blas_name = blasBackendToString(blas);
    std::cerr << "Detected BLAS: " << blas_name << std::endl;

    std::string repo_root = getRepoRoot();
    std::string output_path = getOutputPath(model, device, max_seq_length, batch_size);
    std::string model_path = getTorchscriptModelPath(model, repo_root);
    std::string token_path = getInputTokensPath(model, repo_root);

    std::cout << "Repo root: " << repo_root << std::endl;
    std::cout << "Benchmarking " << model << " on " << device << ", seq len " << max_seq_length
              << ", batch size " << batch_size << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto tokenized_inputs = readTokenizedInputs(token_path, max_seq_length);
    std::cout << "Loaded " << tokenized_inputs.size() << " tokenized inputs" << std::endl;
    std::cout << "Sample input_ids shape: " << tokenized_inputs[0].input_ids.sizes()
              << ", attention mask shape: " << tokenized_inputs[0].attention_mask.sizes() << "\n";

    SentenceEmbedder embedder(model_path, use_mps);

    std::vector<double> run_times;
    torch::Tensor embeddings;

    for (unsigned long i = 0; i < num_runs; ++i) {
        std::cout << "Run " << (i + 1) << "/" << num_runs << std::endl;
        auto start_run = std::chrono::high_resolution_clock::now();

        embeddings = embedder.embed_batch(tokenized_inputs, batch_size);

        auto end_run = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_run - start_run);
        run_times.push_back(duration.count());
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_total - start_total);
    double total_time = total_duration.count();

    // Calculate num_tokens (count non-padding tokens)
    long num_tokens = 0;
    for (const auto& input : tokenized_inputs) {
        auto attention_mask = input.attention_mask;
        num_tokens += torch::sum(attention_mask).item<int>();
    }

    // Create benchmark result JSON with BLAS backend in contestant name
    std::string contestant_name = "libtorch-ts-" + blas_name;
    json benchmark_result = {{"contestant", contestant_name},
                             {"language", "cpp"},
                             {"os", "macos"},
                             {"model", model},
                             {"device", device},
                             {"max_seq_length", max_seq_length},
                             {"batch_size", batch_size},
                             {"total_time", total_time},
                             {"run_times", run_times},
                             {"num_tokens", num_tokens}};

    // Write benchmark result
    std::filesystem::create_directories("output");
    std::ofstream benchmark_file("output/benchmark_results.jsonl", std::ios::app);
    benchmark_file << benchmark_result.dump() << std::endl;
    benchmark_file.close();

    writeEmbeddings(embeddings, output_path);

    std::cout << "Benchmark complete!" << std::endl;
    return 0;
}
