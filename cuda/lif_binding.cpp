#include <torch/extension.h>

// we declare CUDA kernels here

std::vector<<torch::Tensor> lif_forward_cuda(
    torch::Tensor input, torch::Tensor voltage,
    float beta, float threshold);

std::vector<torch::Tensor> lif_backward_cuda(
    torch::Tensor grad_spikes, torch::Tensor voltage,
    float beta, float threshold);

// expose to python

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lif_forward", &lif_forward_cuda, "LIF Forward (CUDA)");
    m.def("lif_backward", &lif_backward_cuda, "LIF Backward (CUDA)");
}
