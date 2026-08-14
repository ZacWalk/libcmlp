
// Multilayer perceptron implementation with forward pass, backpropagation, and AVX2-accelerated math helpers.

#include "nn.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "activation.h"

namespace
{
#if defined(__AVX2__)
xfloat horizontal_sum_avx(__m256 value)
{
	const __m128 low = _mm256_castps256_ps128(value);
	const __m128 high = _mm256_extractf128_ps(value, 1);
	__m128 sum128 = _mm_add_ps(low, high);
	sum128 = _mm_hadd_ps(sum128, sum128);
	sum128 = _mm_hadd_ps(sum128, sum128);
	return _mm_cvtss_f32(sum128);
}
#endif

xfloat dot_product(const xfloat* lhs, const xfloat* rhs, const int count)
{
	xfloat accumulator = 0.0f;
	int index = 0;

#if defined(__AVX2__)
	// Two independent accumulators to keep the FMA pipeline fed.
	__m256 sum_a = _mm256_setzero_ps();
	__m256 sum_b = _mm256_setzero_ps();
	for (; index + 16 <= count; index += 16)
	{
		sum_a = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), sum_a);
		sum_b = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 8), _mm256_loadu_ps(rhs + index + 8), sum_b);
	}
	for (; index + 8 <= count; index += 8)
	{
		sum_a = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), sum_a);
	}
	accumulator = horizontal_sum_avx(_mm256_add_ps(sum_a, sum_b));
#endif

	for (; index < count; index += 1)
	{
		accumulator += lhs[index] * rhs[index];
	}

	return accumulator;
}

void scaled_accumulate(xfloat* destination, const xfloat* input, const int count, const xfloat scale)
{
	int index = 0;

#if defined(__AVX2__)
	const __m256 scale_vector = _mm256_set1_ps(scale);
	for (; index + 8 <= count; index += 8)
	{
		const __m256 accumulated = _mm256_fmadd_ps(scale_vector, _mm256_loadu_ps(input + index), _mm256_loadu_ps(destination + index));
		_mm256_storeu_ps(destination + index, accumulated);
	}
#endif

	for (; index < count; index += 1)
	{
		destination[index] += scale * input[index];
	}
}

void apply_momentum_update(xfloat* weights, xfloat* velocity, xfloat* gradients, const std::size_t count, const xfloat momentum, const xfloat learning_rate)
{
	std::size_t index = 0;

#if defined(__AVX2__)
	const __m256 momentum_vector = _mm256_set1_ps(momentum);
	const __m256 learning_rate_vector = _mm256_set1_ps(learning_rate);
	const __m256 zero_vector = _mm256_setzero_ps();
	for (; index + 8 <= count; index += 8)
	{
		const __m256 velocity_vector = _mm256_loadu_ps(velocity + index);
		const __m256 gradient_vector = _mm256_loadu_ps(gradients + index);
		const __m256 updated_velocity = _mm256_fnmadd_ps(learning_rate_vector, gradient_vector, _mm256_mul_ps(momentum_vector, velocity_vector));
		_mm256_storeu_ps(velocity + index, updated_velocity);
		_mm256_storeu_ps(weights + index, _mm256_add_ps(_mm256_loadu_ps(weights + index), updated_velocity));
		_mm256_storeu_ps(gradients + index, zero_vector);
	}
#endif

	for (; index < count; index += 1)
	{
		velocity[index] = momentum * velocity[index] - learning_rate * gradients[index];
		weights[index] += velocity[index];
		gradients[index] = 0.0f;
	}
}
}

xfloat* nn::activation_ptr(const std::size_t layer)
{
	return activations.data() + layers[layer].activation_offset;
}

const xfloat* nn::activation_ptr(const std::size_t layer) const
{
	return activations.data() + layers[layer].activation_offset;
}

xfloat* nn::delta_ptr(const std::size_t layer)
{
	return deltas.data() + layers[layer].delta_offset;
}

const xfloat* nn::delta_ptr(const std::size_t layer) const
{
	return deltas.data() + layers[layer].delta_offset;
}

xfloat* nn::weight_ptr(const std::size_t layer)
{
	return weights.data() + connections[layer].weight_offset;
}

const xfloat* nn::weight_ptr(const std::size_t layer) const
{
	return weights.data() + connections[layer].weight_offset;
}

xfloat* nn::gradient_ptr(const std::size_t layer)
{
	return gradient_accumulators.data() + connections[layer].weight_offset;
}

xfloat* nn::velocity_ptr(const std::size_t layer)
{
	return velocity.data() + connections[layer].weight_offset;
}

sample_metrics nn::evaluate_output(const xfloat* Y, const int dim, const bool write_output_delta)
{
	const std::size_t last_layer = layers.size() - 1;
	const auto* output = activation_ptr(last_layer);
	auto* output_delta = write_output_delta ? delta_ptr(last_layer) : nullptr;
	int label = 0;
	int target_label = 0;
	xfloat max_val = output[0];

	for (int neuron = 0; neuron < dim; neuron += 1)
	{
		const xfloat prediction = output[neuron];
		if (Y[neuron] > 0.0f)
		{
			target_label = neuron;
		}
		if (write_output_delta)
		{
			output_delta[neuron] = prediction - Y[neuron];
		}
		if (prediction > max_val)
		{
			max_val = prediction;
			label = neuron;
		}
	}

	sample_metrics metrics{};
	metrics.loss = -std::log(std::max(output[target_label], std::numeric_limits<xfloat>::min()));
	metrics.correct = label == target_label ? 1u : 0u;
	return metrics;
}

sample_metrics nn::accumulate_gradients(const xfloat* X, const xfloat* Y, const int dim)
{
	load_input(X);
	forward();
	const auto metrics = evaluate_output(Y, dim, true);
	back_propagation();
	accumulate_weight_gradients();
	return metrics;
}

void nn::apply_batch(const std::size_t batch_size)
{
	if (batch_size == 0)
	{
		throw std::invalid_argument("Batch size must be positive");
	}

	// Gradients are averaged over the batch by folding 1/batch_size into the step.
	const xfloat batch_learning_rate = learning_rate / static_cast<xfloat>(batch_size);
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		apply_momentum_update(weight_ptr(layer), velocity_ptr(layer), gradient_ptr(layer), connections[layer].weight_count, momentum, batch_learning_rate);
	}
}

sample_metrics nn::evaluate_sample(const xfloat* X, const xfloat* Y, const int dim)
{
	load_input(X);
	forward();
	return evaluate_output(Y, dim, false);
}

void nn::back_propagation(void)
{
	const std::size_t last_layer = layers.size() - 1;

	for (std::size_t layer = last_layer - 1; layer > 0; layer -= 1)
	{
		const int width = layers[layer].width;
		const auto& next_connection = connections[layer];
		const auto* next_weights = weight_ptr(layer);
		const auto* next_delta = delta_ptr(layer + 1);
		const auto* current_activation = activation_ptr(layer);
		auto* current_delta = delta_ptr(layer);

		// Walk the weight matrix row-wise so the accumulation stays contiguous and vectorizable.
		// The bias column is skipped because a constant input has no delta to propagate.
		std::fill_n(current_delta, width, 0.0f);
		for (int next_neuron = 0; next_neuron < next_connection.output_width; next_neuron += 1)
		{
			scaled_accumulate(current_delta, next_weights + static_cast<std::size_t>(next_neuron) * next_connection.input_width, width, next_delta[next_neuron]);
		}

		for (int neuron = 0; neuron < width; neuron += 1)
		{
			current_delta[neuron] *= sig_derivative(current_activation[neuron]);
		}
	}
}

void nn::accumulate_weight_gradients(void)
{
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		const auto& connection = connections[layer];
		auto* layer_gradients = gradient_ptr(layer);
		const auto* layer_input = activation_ptr(layer);
		const auto* layer_delta = delta_ptr(layer + 1);

		for (int neuron = 0; neuron < connection.output_width; neuron += 1)
		{
			auto* neuron_gradients = layer_gradients + static_cast<std::size_t>(neuron) * connection.input_width;
			scaled_accumulate(neuron_gradients, layer_input, connection.input_width, layer_delta[neuron]);
		}
	}
}

void nn::forward(void)
{
	for (std::size_t layer = 1; layer < layers.size(); layer += 1)
	{
		const auto& current_layer = layers[layer];
		const auto& current_connection = connections[layer - 1];
		const auto* previous_activation = activation_ptr(layer - 1);
		auto* current_activation = activation_ptr(layer);
		const auto* layer_weights = weight_ptr(layer - 1);

		if (current_layer.has_bias)
		{
			for (int neuron = 0; neuron < current_layer.width; neuron += 1)
			{
				const auto* neuron_weights = layer_weights + static_cast<std::size_t>(neuron) * current_connection.input_width;
				current_activation[neuron] = sigmoid(dot_product(neuron_weights, previous_activation, current_connection.input_width));
			}
			current_activation[current_layer.width] = 1.0f;
			continue;
		}

		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			const auto* neuron_weights = layer_weights + static_cast<std::size_t>(neuron) * current_connection.input_width;
			current_activation[neuron] = dot_product(neuron_weights, previous_activation, current_connection.input_width);
		}

		// Output layer: softmax, shifted by the maximum logit for numerical stability.
		const xfloat max_logit = *std::max_element(current_activation, current_activation + current_layer.width);
		xfloat denominator = 0.0f;
		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			current_activation[neuron] = std::exp(current_activation[neuron] - max_logit);
			denominator += current_activation[neuron];
		}

		const xfloat inverse_denominator = 1.0f / denominator;
		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			current_activation[neuron] *= inverse_denominator;
		}
	}
}

void nn::compile(const nn_config& config)
{
	const auto& l = config.layers;
	if (l.size() < 2)
	{
		throw std::invalid_argument("Network must contain at least an input and output layer");
	}

	for (const int width : l)
	{
		if (width <= 0)
		{
			throw std::invalid_argument("Layer widths must be positive");
		}
	}

	if (config.learning_rate <= 0.0f)
	{
		throw std::invalid_argument("Learning rate must be positive");
	}

	if (config.momentum < 0.0f || config.momentum >= 1.0f)
	{
		throw std::invalid_argument("Momentum must be in [0, 1)");
	}

	layers.assign(l.size(), {});
	connections.assign(l.size() - 1, {});
	learning_rate = config.learning_rate;
	momentum = config.momentum;

	std::size_t activation_size = 0;
	for (std::size_t layer = 0; layer < layers.size(); layer += 1)
	{
		const bool has_bias = layer + 1 < l.size();
		layers[layer].width = l[layer];
		layers[layer].has_bias = has_bias;
		layers[layer].activation_width = l[layer] + (has_bias ? 1 : 0);
		layers[layer].activation_offset = activation_size;
		activation_size += static_cast<std::size_t>(layers[layer].activation_width);
	}
	activations.assign(activation_size, 0.0f);

	std::size_t delta_size = 0;
	for (std::size_t layer = 1; layer < layers.size(); layer += 1)
	{
		layers[layer].delta_offset = delta_size;
		delta_size += static_cast<std::size_t>(layers[layer].width);
	}
	deltas.assign(delta_size, 0.0f);

	std::size_t weight_size = 0;
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		auto& connection = connections[layer];
		connection.input_width = layers[layer].activation_width;
		connection.output_width = layers[layer + 1].width;
		connection.weight_offset = weight_size;
		connection.weight_count = static_cast<std::size_t>(connection.output_width) * static_cast<std::size_t>(connection.input_width);
		weight_size += connection.weight_count;
	}
	weights.resize(weight_size);
	gradient_accumulators.assign(weight_size, 0.0f);
	velocity.assign(weight_size, 0.0f);

	std::mt19937 gen(config.seed != 0 ? config.seed : std::random_device{}());
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		const auto& connection = connections[layer];
		const xfloat limit = std::sqrt(6.0f / static_cast<xfloat>(connection.input_width + connection.output_width));
		std::uniform_real_distribution<xfloat> dist(-limit, limit);
		auto* layer_weights = weight_ptr(layer);
		for (std::size_t weight_index = 0; weight_index < connection.weight_count; weight_index += 1)
		{
			layer_weights[weight_index] = dist(gen);
		}
	}
}

void nn::scale_learning_rate(const xfloat factor)
{
	learning_rate *= factor;
}

void nn::load_input(const xfloat* X)
{
	if (layers.empty())
	{
		throw std::logic_error("Network must be compiled before use");
	}

	auto* input = activation_ptr(0);
	std::copy_n(X, static_cast<std::size_t>(layers[0].width), input);
	if (layers[0].has_bias)
	{
		input[layers[0].width] = 1.0f;
	}
}

void nn::summary(void) const
{
	int layer_number = 0;

	std::cout << "\n\nNeural Network Summary:\t\t[hidden := Sigmoid, output := Softmax]" << std::endl << std::endl;

	for (std::size_t layer = 0; layer < layers.size(); layer += 1)
	{
		std::cout << "Layer " << ++layer_number << "\t" << std::setw(4) << layers[layer].width << " neurons";
		if (layers[layer].has_bias)
		{
			std::cout << " + bias";
		}
		std::cout << "\n";
	}

	std::cout << "Parameters\t" << weights.size() << " trainable weights\n";
	std::cout << "Learning Rate\t" << learning_rate << "\n";
	std::cout << "Momentum\t" << momentum << "\n";
}