#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * One CUDA thread = one neuron at one timestep.
 *
 * Implements the snnTorch ``Leaky`` subtract-on-reset dynamics:
 *
 *     v_pre  = beta * voltage_in + input
 *     spike  = (v_pre >= threshold) ? 1 : 0
 *     v_post = v_pre - spike * threshold   (spike treated as detached)
 *
 * The pre-reset voltage v_pre is not returned here; the autograd wrapper
 * recomputes it from (input, voltage_in, beta) at backward time, which is
 * one fused-multiply-add per element and avoids an extra allocation.
 */
__global__ void lif_forward_kernel(
    const float* __restrict__ input,        // [batch * neurons]
    const float* __restrict__ voltage_in,   // [batch * neurons]
    float* __restrict__ voltage_out,        // [batch * neurons]
    float* __restrict__ spikes,             // [batch * neurons]
    float beta,
    float threshold,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float v = beta * voltage_in[idx] + input[idx];
    float spike = (v >= threshold) ? 1.0f : 0.0f;
    voltage_out[idx] = v - spike * threshold;
    spikes[idx] = spike;
}

std::vector<torch::Tensor> lif_forward_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float beta,
    float threshold
) {
    TORCH_CHECK(input.is_cuda(),   "input must be a CUDA tensor");
    TORCH_CHECK(voltage.is_cuda(), "voltage must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(),   "input must be contiguous");
    TORCH_CHECK(voltage.is_contiguous(), "voltage must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,   "input must be float32");
    TORCH_CHECK(voltage.scalar_type() == torch::kFloat32, "voltage must be float32");
    TORCH_CHECK(input.sizes() == voltage.sizes(),
                "input and voltage must have the same shape");

    const int total = static_cast<int>(input.numel());
    auto voltage_out = torch::empty_like(voltage);
    auto spikes = torch::empty_like(input);

    constexpr int THREADS = 256;
    const int blocks = (total + THREADS - 1) / THREADS;

    lif_forward_kernel<<<blocks, THREADS>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        voltage_out.data_ptr<float>(),
        spikes.data_ptr<float>(),
        beta,
        threshold,
        total
    );

    return {spikes, voltage_out};
}
