// Neural network declarations for the multilayer perceptron, its configuration, and training APIs.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.h"

struct sample_metrics
{
    xfloat loss = 0.0f;
    std::size_t correct = 0;
};

struct nn_config
{
    std::vector<int> layers;
    xfloat learning_rate = LEARNING_RATE;
    xfloat momentum = MOMENTUM;
    std::uint32_t seed = 0; // 0 draws a nondeterministic seed
};

// Multi Layer Perceptron
class nn
{
public:
    void compile(const nn_config& config);
    sample_metrics accumulate_gradients(const xfloat* X, const xfloat* Y, int dim);
    void apply_batch(std::size_t batch_size);
    sample_metrics evaluate_sample(const xfloat* X, const xfloat* Y, int dim);
    void scale_learning_rate(xfloat factor);
    void summary(void) const;

private:
    struct layer_descriptor
    {
        int width = 0;
        int activation_width = 0;
        bool has_bias = false;
        std::size_t activation_offset = 0;
        std::size_t delta_offset = 0;
    };

    struct connection_descriptor
    {
        int input_width = 0;
        int output_width = 0;
        std::size_t weight_offset = 0;
        std::size_t weight_count = 0;
    };

    std::vector<layer_descriptor> layers;
    std::vector<connection_descriptor> connections;
    std::vector<xfloat> activations;
    std::vector<xfloat> deltas;
    std::vector<xfloat> weights;
    std::vector<xfloat> gradient_accumulators;
    std::vector<xfloat> velocity;
    xfloat learning_rate = LEARNING_RATE;
    xfloat momentum = MOMENTUM;

    void load_input(const xfloat* X);
    void forward(void);
    void back_propagation(void);
    void accumulate_weight_gradients(void);
    sample_metrics evaluate_output(const xfloat* Y, int dim, bool write_output_delta);

    xfloat* activation_ptr(std::size_t layer);
    const xfloat* activation_ptr(std::size_t layer) const;
    xfloat* delta_ptr(std::size_t layer);
    const xfloat* delta_ptr(std::size_t layer) const;
    xfloat* weight_ptr(std::size_t layer);
    const xfloat* weight_ptr(std::size_t layer) const;
    xfloat* gradient_ptr(std::size_t layer);
    xfloat* velocity_ptr(std::size_t layer);
};