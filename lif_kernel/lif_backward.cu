/**
 * Backward pass for the LIF neuron using a fast-sigmoid surrogate
 * gradient. Computes d_loss / d_v_pre where v_pre = beta * mem_prev + input
 * (the pre-reset membrane potential at the current step).
 *
 * The caller is responsible for combining this with the gradient flowing
 * back through mem_new (the post-reset state). See cuda/lif_function.py.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fast-sigmoid surrogate: sg(x) = 1 / (1 + |x| * slope)^2
// Matches snnTorch's surrogate.fast_sigmoid(slope=...) at the scalar level.
__device__ __forceinline__ float surrogate_grad(
    float v_pre, float threshold, float slope
) {
    float x = v_pre - threshold;
    float denom = 1.0f + slope * fabsf(x);
    return 1.0f / (denom * denom);
}

__global__ void lif_backward_kernel(
    const float* __restrict__ grad_spikes,  // [batch * neurons]
    const float* __restrict__ v_pre,        // [batch * neurons]
    float* __restrict__ grad_v_pre,         // [batch * neurons] -- output
    float threshold,
    float slope,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float sg = surrogate_grad(v_pre[idx], threshold, slope);
    grad_v_pre[idx] = grad_spikes[idx] * sg;
}

torch::Tensor lif_backward_cuda(
    torch::Tensor grad_spikes,
    torch::Tensor v_pre,
    float threshold,
    float surrogate_slope
) {
    TORCH_CHECK(grad_spikes.is_cuda(), "grad_spikes must be a CUDA tensor");
    TORCH_CHECK(v_pre.is_cuda(),       "v_pre must be a CUDA tensor");
    TORCH_CHECK(grad_spikes.is_contiguous(), "grad_spikes must be contiguous");
    TORCH_CHECK(v_pre.is_contiguous(),       "v_pre must be contiguous");
    TORCH_CHECK(grad_spikes.scalar_type() == torch::kFloat32, "grad_spikes must be float32");
    TORCH_CHECK(v_pre.scalar_type() == torch::kFloat32,       "v_pre must be float32");
    TORCH_CHECK(grad_spikes.sizes() == v_pre.sizes(),
                "grad_spikes and v_pre must have the same shape");

    const int total = static_cast<int>(grad_spikes.numel());
    auto grad_v_pre = torch::empty_like(v_pre);

    constexpr int THREADS = 256;
    const int blocks = (total + THREADS - 1) / THREADS;

    lif_backward_kernel<<<blocks, THREADS>>>(
        grad_spikes.data_ptr<float>(),
        v_pre.data_ptr<float>(),
        grad_v_pre.data_ptr<float>(),
        threshold,
        surrogate_slope,
        total
    );

    return grad_v_pre;
}
