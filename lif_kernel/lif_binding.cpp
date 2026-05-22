#include <torch/extension.h>
#include <vector>

// Forward declarations of the CUDA launch functions defined in .cu files.

std::vector<torch::Tensor> lif_forward_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float beta,
    float threshold);

torch::Tensor lif_backward_cuda(
    torch::Tensor grad_spikes,
    torch::Tensor v_pre,
    float threshold,
    float surrogate_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lif_forward", &lif_forward_cuda, "LIF forward (CUDA)");
    m.def("lif_backward", &lif_backward_cuda, "LIF backward surrogate (CUDA)");
}
