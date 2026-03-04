#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// each CUDA thread handles ONE neuron at ONE timestamp

/**
* Behavior:
* 	Kernel for calculating spikes and outward facing voltage for a layer 
* 	of LIFs in the SRNN. Uses the LIF formula to calculate the voltage,
* 	and spikes if voltage is over a certain threshold.
*/
__global__ void lif_forward_kernel(
	const float* __restrict__ input,	// [batch, neurons]
	const float* __restrict__ voltage_in,   // [batch, neurons] -- volt from prev
	float* __restrict__ voltage_out,	// [batch, neurons] -- new state
	float* __restrict__ spikes,		// [batch, neurons] -- output 0 / 1
	float beta,		// leak factor
	float threshold, 	// firing threshold
	int batch_size,
	int num_neurons
) {
	// figure out which neuron/batch this thread is responsible for
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch_size * num_neurons;

	if (idx >= total) return;
	
	// calculate voltage for neuron
	float v = beta * voltage_in[idx] + input[idx];
	
	// fire if above threshold
	float spike = (v >= threshold) ? 1.0f : 0.0f;

	// reset voltage if fired
	voltage_out[idx] = v - spike * threshold;
	spikes[idx] = spike;
}

// pytorch bindings function

/**
* Behavior:
* 	A Kernel Launch function for lif_forward. Calculates the spikes and voltages
* 	from the current layer given a controlled input, the voltage from the previous
* 	layer, a beta leak factor, and a firing threshold
*/

std::vector<torch::Tensor> lif_forward_cuda(
	torch::Tensor input,
	torch::Tensor voltage,
	float beta,
	float threshold
) {
	// use auto type deduction because torch--kinda cool
	auto batch_size = input.size(0);
	auto num_neurons = input.size(1);
	int total = batch_size * num_neurons;

	auto voltage_out = torch::zeros_like(voltage);
	auto spikes = torch::zeros_like(input);

	constexpr int THREADS = 256;
	const int blocks = (total + THREADS - 1) / THREADS;

	lif_forward_kernel<<<blocks, THREADS>>>(
		input.data_ptr<float>(),
		voltage.data_ptr<float>(),
		voltage_out.data_ptr<float>(),
		spikes.data_ptr<float>(),
		beta,
		threshold,
		batch_size,
		num_neurons
	);

	return {spikes, voltage_out};

}
