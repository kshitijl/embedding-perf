#include <torch/torch.h>

#include <iostream>

int main() {
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
    return 0;
}
