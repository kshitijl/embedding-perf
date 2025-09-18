#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: main <path-to-torchscript>\n";
        return 1;
    }

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
    }
    if (torch::mps::is_available()) {
        std::cout << "MPS is available!" << std::endl;
        // Your code to run on MPS
        torch::Device device(torch::kMPS);
        torch::Tensor tensor = torch::ones({2, 2}, device);
        std::cout << tensor << std::endl;
    } else {
        std::cout << "MPS not available, using CPU." << std::endl;
        // Your code to run on CPU
        torch::Device device(torch::kCPU);
        torch::Tensor tensor = torch::ones({2, 2}, device);
        std::cout << tensor << std::endl;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading model\n";
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
