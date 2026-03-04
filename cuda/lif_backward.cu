/**
* This is the backward pass kernel that uses a Surrogate Gradient. 
* Integrated with Pytorch
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Surrogate gradient: we gotta use the derivative of a fast sigmoid
// Real spike gradient is zero everywhere, but we approximate it

__device__ float surrogate_grad(float voltage, float threshold) {
    float x = voltage - threshold;
    // fast sigmoid surrogage: 1 / (1 + |x| * scale)^2
    float scale = 10.f;
    float denom = 1.0f + scale * fabsf(x);
    return 1.0f / (denom * denom);
}

/**
* Behavior:
*   Backwards kernel that calculates the gradiant for the voltage
*/

__global__ void lif_backward_kernel(
    const float* __restrict__ grad_spikes,  // gradient flowing in from above    
    const float* __restrict__ voltage,      // saved voltage from forward pass
    float* __restrict__ grad_input,         // gradient w.r.t input
    float* __restrict__ grad_voltage,       // gradient w.r.t voltage state
    float beta,
    float threshold,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Chain rule: d_loss/d_input = d_loss/d_spike * d_spike/d_voltage
    float sg = surrogate_grad(voltage[idx], threshold);
    grad_input[idx] = grad_spikes[idx] * sg;

    // gradient also flows through leak path (beta * V_prev)
    grad_voltage[idx] = grad_spikes[idx] * sg * beta;
}


/**
* Behavior:
*   Launches lif_backward_kernel by setting up arguments +
*   integrating with Pytorch
*/
std::vector<torch::Tensor> lif_backward_cuda(
    torch::Tensor grad_spikes,
    torch::Tensor voltage,
    float beta,
    float threshold
) {
    int total = grad_spikes.numel();
    auto grad_input = torch::zeros_like(grad_spikes);
    auto grad_voltage = torch::zeros_like(voltage);

    constexpr int THREADS = 256;
    const int blocks = (THREADS + total - 1) / THREADS;

    lif_backward_kernel<<THREADS, blocks>>>(
        grad_spikes.data_ptr<float>(),
        voltage.data_ptr<float>(),
        grad_input,
        grad_voltage,
        beta,
        threshold,
        total
    );

    return {grad_input, grad_voltage};
}
