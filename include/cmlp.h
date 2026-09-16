#ifndef CMLP_H
#define CMLP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cmlp_status {
    CMLP_OK = 0,
    CMLP_INVALID_ARGUMENT,
    CMLP_OUT_OF_MEMORY,
    CMLP_INVALID_STATE,
    CMLP_IO_ERROR,
    CMLP_INVALID_FORMAT,
    CMLP_NUMERIC_ERROR
} cmlp_status;

typedef enum cmlp_activation {
    CMLP_LINEAR = 0,
    CMLP_SIGMOID,
    CMLP_RELU,
    CMLP_TANH,
    CMLP_SOFTMAX
} cmlp_activation;

typedef enum cmlp_optimizer {
    CMLP_SGD = 0,
    CMLP_ADAM
} cmlp_optimizer;

typedef struct cmlp_config {
    const int *layers;
    size_t layer_count;
    cmlp_activation hidden_activation;
    cmlp_activation output_activation;
    cmlp_optimizer optimizer;
    float learning_rate;
    float momentum;
    float beta1;
    float beta2;
    float epsilon;
    float gradient_clip;
    uint32_t seed;
} cmlp_config;

typedef struct cmlp_metrics {
    float loss;
    size_t correct;
} cmlp_metrics;

typedef struct cmlp_model cmlp_model;

/* Config borrows layers during create only. Zero seed requests a fresh seed.
 * Defaults are sigmoid/softmax, SGD+momentum; gradient_clip=0 disables clipping. */
cmlp_config cmlp_default_config(void);
const char *cmlp_status_string(cmlp_status status);
cmlp_status cmlp_create(const cmlp_config *config, cmlp_model **out);
void cmlp_destroy(cmlp_model *model);
size_t cmlp_input_size(const cmlp_model *model);
size_t cmlp_output_size(const cmlp_model *model);
size_t cmlp_parameter_count(const cmlp_model *model);

/* Rows are contiguous. The model retains the most recent forward batch for
 * backward; output may be NULL when only the retained activations are needed.
 * A model is mutable and must not be used concurrently. */
cmlp_status cmlp_forward(cmlp_model *model, const float *input, size_t samples, float *output);
/* grad_output is dLoss/dOutput, not dLoss/dLogit; gradients accumulate until step.
 * Supply unaveraged per-sample derivatives: step averages over accumulated rows.
 * samples must equal the retained batch size. Successful backward consumes it.
 * A failed forward invalidates the retained batch. Numeric overflow while
 * accumulating gradients discards all pending gradients, never parameters.
 * A failed step leaves parameters, gradients and optimizer state unchanged. */
cmlp_status cmlp_backward(cmlp_model *model, const float *grad_output, size_t samples);
cmlp_status cmlp_step(cmlp_model *model);
cmlp_status cmlp_zero_grad(cmlp_model *model);

/* Categorical cross-entropy helpers require softmax output and a probability
 * target of output_size floats (sum within 1e-5 of one). Metrics may be NULL.
 * train_sample accumulates one row; call step to update the parameters. */
cmlp_status cmlp_train_sample(cmlp_model *model, const float *input,
    const float *target, cmlp_metrics *metrics);
cmlp_status cmlp_evaluate_sample(cmlp_model *model, const float *input,
    const float *target, cmlp_metrics *metrics);
cmlp_status cmlp_set_learning_rate(cmlp_model *model, float rate);
cmlp_status cmlp_scale_learning_rate(cmlp_model *model, float factor);
float cmlp_learning_rate(const cmlp_model *model);

/* Copy parameters between identically shaped/activated networks, resetting the
 * destination gradients and optimizer state (DQN target-network semantics). */
cmlp_status cmlp_copy_from(cmlp_model *destination, const cmlp_model *source);
/* Versioned, checksummed binary32 checkpoints retain optimizer configuration,
 * moments, update count and pending gradients, not the forward batch.
 * Save overwrites path; a write failure may leave a partial file.
 * On create/load failure *out is unchanged; success returns a new owned model
 * (destroy it with cmlp_destroy). Files from other implementations are rejected. */
cmlp_status cmlp_save(const cmlp_model *model, const char *path);
cmlp_status cmlp_load(const char *path, cmlp_model **out);

#ifdef __cplusplus
}
#endif
#endif
