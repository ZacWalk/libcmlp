#include "cmlp.h"
#include "common.h"
#include "activation.h"
#include "random.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct nn_layer {
    size_t in, out;
    size_t weight, bias;
    size_t input, output;
} nn_layer;

struct cmlp_model {
    cmlp_config config;
    int *widths;
    nn_layer *layers;
    size_t count, stride, parameters;
    float *params, *grads, *moment1, *moment2;
    float *next_params, *next_moment1, *next_moment2;
    float *act, *delta, *logits;
    size_t capacity, batch, accumulated;
    double gradient_bound;
    float softmax_max, softmax_sum;
    uint64_t steps;
};

static int checked_add(size_t a, size_t b, size_t *result)
{
    if (b > SIZE_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int checked_mul(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

static int float_count(size_t count)
{
    return count <= SIZE_MAX / sizeof(float) && count <= PTRDIFF_MAX / sizeof(float);
}

static int all_finite(const float *values, size_t count)
{
    size_t i = 0;
#if defined(__AVX2__)
    __m256i maximum = _mm256_setzero_si256();
    const __m256i magnitude = _mm256_set1_epi32(INT32_MAX);
    for (; count - i >= 8; i += 8) {
        const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(values + i));
        maximum = _mm256_max_epu32(maximum, _mm256_and_si256(bits, magnitude));
    }
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(maximum,
            _mm256_set1_epi32(INT32_C(0x7f7fffff)))) != 0) return 0;
#endif
    for (; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static float maximum_magnitude(const float *values, size_t count)
{
    float maximum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    const __m256i magnitude = _mm256_set1_epi32(INT32_MAX);
    __m256i max_vector = _mm256_setzero_si256();
    __m128i reduced;
    uint32_t bits;
    for (; count - i >= 8; i += 8)
        max_vector = _mm256_max_epu32(max_vector, _mm256_and_si256(magnitude,
            _mm256_castps_si256(_mm256_loadu_ps(values + i))));
    reduced = _mm_max_epu32(_mm256_castsi256_si128(max_vector), _mm256_extracti128_si256(max_vector, 1));
    reduced = _mm_max_epu32(reduced, _mm_shuffle_epi32(reduced, _MM_SHUFFLE(1, 0, 3, 2)));
    reduced = _mm_max_epu32(reduced, _mm_shuffle_epi32(reduced, _MM_SHUFFLE(2, 3, 0, 1)));
    bits = (uint32_t)_mm_cvtsi128_si32(reduced);
    memcpy(&maximum, &bits, sizeof(maximum));
#endif
    for (; i < count; ++i) {
        const float value = fabsf(values[i]);
        if (value > maximum) maximum = value;
    }
    return maximum;
}

#if defined(__AVX2__)
static NN_FORCEINLINE float horizontal_sum(__m256 value)
{
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
        _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static NN_FORCEINLINE __m256 multiply_add(__m256 a, __m256 b, __m256 c)
{
#if defined(__FMA__) || defined(_MSC_VER)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

static NN_FORCEINLINE float dot_product(const float *a, const float *b, size_t count)
{
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 sum_a = _mm256_setzero_ps(), sum_b = _mm256_setzero_ps();
    for (; count - i >= 16; i += 16) {
        sum_a = multiply_add(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum_a);
        sum_b = multiply_add(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum_b);
    }
    for (; count - i >= 8; i += 8)
        sum_a = multiply_add(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum_a);
    sum = horizontal_sum(_mm256_add_ps(sum_a, sum_b));
#endif
    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;
}

static NN_FORCEINLINE void scaled_accumulate(float *restrict dst, const float *restrict src,
    size_t count, float scale)
{
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 factor = _mm256_set1_ps(scale);
    for (; count - i >= 8; i += 8)
        _mm256_storeu_ps(dst + i, multiply_add(factor,
            _mm256_loadu_ps(src + i), _mm256_loadu_ps(dst + i)));
#endif
    for (; i < count; ++i) dst[i] += scale * src[i];
}

static NN_FORCEINLINE int checked_accumulate(float *restrict dst, const float *restrict src,
    size_t count, float scale)
{
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 factor = _mm256_set1_ps(scale);
    const __m256i magnitude = _mm256_set1_epi32(INT32_MAX);
    __m256i maximum = _mm256_setzero_si256();
    for (; count - i >= 8; i += 8) {
        const __m256 next = multiply_add(factor,
            _mm256_loadu_ps(src + i), _mm256_loadu_ps(dst + i));
        maximum = _mm256_max_epu32(maximum,
            _mm256_and_si256(_mm256_castps_si256(next), magnitude));
        _mm256_storeu_ps(dst + i, next);
    }
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(maximum,
            _mm256_set1_epi32(INT32_C(0x7f7fffff)))) != 0) return 0;
#endif
    for (; i < count; ++i) {
        dst[i] += scale * src[i];
        if (!isfinite(dst[i])) return 0;
    }
    return 1;
}

cmlp_config cmlp_default_config(void)
{
    cmlp_config config;
    memset(&config, 0, sizeof(config));
    config.hidden_activation = CMLP_SIGMOID;
    config.output_activation = CMLP_SOFTMAX;
    config.optimizer = CMLP_SGD;
    config.learning_rate = LEARNING_RATE;
    config.momentum = MOMENTUM;
    config.beta1 = ADAM_BETA1;
    config.beta2 = ADAM_BETA2;
    config.epsilon = ADAM_EPSILON;
    return config;
}

const char *cmlp_status_string(cmlp_status status)
{
    switch (status) {
    case CMLP_OK: return "success";
    case CMLP_INVALID_ARGUMENT: return "invalid argument";
    case CMLP_OUT_OF_MEMORY: return "out of memory";
    case CMLP_INVALID_STATE: return "invalid model state";
    case CMLP_IO_ERROR: return "I/O error";
    case CMLP_INVALID_FORMAT: return "invalid checkpoint format";
    case CMLP_NUMERIC_ERROR: return "nonfinite value or numeric overflow";
    default: return "unknown status";
    }
}

static int valid_config(const cmlp_config *config)
{
    return config != NULL && config->layers != NULL && config->layer_count >= 2
        && config->layer_count <= UINT32_MAX
        && config->layer_count <= PTRDIFF_MAX / sizeof(nn_layer)
        && config->layer_count <= SIZE_MAX / sizeof(nn_layer)
        && config->hidden_activation >= CMLP_LINEAR
        && config->hidden_activation <= CMLP_TANH
        && config->output_activation >= CMLP_LINEAR
        && config->output_activation <= CMLP_SOFTMAX
        && (config->optimizer == CMLP_SGD || config->optimizer == CMLP_ADAM)
        && isfinite(config->learning_rate) && config->learning_rate > 0.0f
        && isfinite(config->momentum) && config->momentum >= 0.0f && config->momentum < 1.0f
        && isfinite(config->beta1) && config->beta1 >= 0.0f && config->beta1 < 1.0f
        && isfinite(config->beta2) && config->beta2 >= 0.0f && config->beta2 < 1.0f
        && isfinite(config->epsilon) && config->epsilon > 0.0f
        && isfinite(config->gradient_clip) && config->gradient_clip >= 0.0f;
}

static cmlp_status ensure_capacity(cmlp_model *model, size_t samples)
{
    size_t count, logit_count;
    float *act, *delta, *logits;
    if (!checked_mul(samples, model->stride, &count) || !float_count(count)
        || !checked_mul(samples, cmlp_output_size(model), &logit_count)
        || !float_count(logit_count)) return CMLP_INVALID_ARGUMENT;
    if (samples <= model->capacity) return CMLP_OK;
    act = (float *)malloc(count * sizeof(float));
    delta = (float *)malloc(count * sizeof(float));
    logits = (float *)malloc(logit_count * sizeof(float));
    if (act == NULL || delta == NULL || logits == NULL) {
        free(act); free(delta); free(logits);
        return CMLP_OUT_OF_MEMORY;
    }
    free(model->act); free(model->delta); free(model->logits);
    model->act = act; model->delta = delta; model->logits = logits;
    model->capacity = samples;
    return CMLP_OK;
}

cmlp_status cmlp_create(const cmlp_config *config, cmlp_model **out)
{
    cmlp_model *model;
    size_t i, parameters = 0, stride = 0;
    nn_rng rng;
    cmlp_status status;
    if (out == NULL || !valid_config(config)) return CMLP_INVALID_ARGUMENT;
    for (i = 0; i < config->layer_count; ++i) {
        size_t weights, connection;
        if (config->layers[i] <= 0
            || !checked_add(stride, (size_t)config->layers[i], &stride))
            return CMLP_INVALID_ARGUMENT;
        if (i == 0) continue;
        if (!checked_mul((size_t)config->layers[i - 1], (size_t)config->layers[i], &weights)
            || !checked_add(weights, (size_t)config->layers[i], &connection)
            || !checked_add(parameters, connection, &parameters))
            return CMLP_INVALID_ARGUMENT;
    }
    if (!float_count(parameters) || !float_count(stride)) return CMLP_INVALID_ARGUMENT;
    model = (cmlp_model *)calloc(1, sizeof(*model));
    if (model == NULL) return CMLP_OUT_OF_MEMORY;
    model->config = *config;
    model->count = config->layer_count - 1;
    model->parameters = parameters;
    model->stride = stride;
    model->widths = (int *)malloc(config->layer_count * sizeof(int));
    model->layers = (nn_layer *)calloc(model->count, sizeof(nn_layer));
    model->params = (float *)calloc(parameters, sizeof(float));
    model->grads = (float *)calloc(parameters, sizeof(float));
    model->moment1 = (float *)calloc(parameters, sizeof(float));
    model->next_params = (float *)malloc(parameters * sizeof(float));
    model->next_moment1 = (float *)malloc(parameters * sizeof(float));
    if (config->optimizer == CMLP_ADAM) {
        model->moment2 = (float *)calloc(parameters, sizeof(float));
        model->next_moment2 = (float *)malloc(parameters * sizeof(float));
    }
    if (model->widths == NULL || model->layers == NULL || model->params == NULL
        || model->grads == NULL || model->moment1 == NULL || model->next_params == NULL
        || model->next_moment1 == NULL || (config->optimizer == CMLP_ADAM
            && (model->moment2 == NULL || model->next_moment2 == NULL))) {
        cmlp_destroy(model);
        return CMLP_OUT_OF_MEMORY;
    }
    memcpy(model->widths, config->layers, config->layer_count * sizeof(int));
    model->config.layers = model->widths;
    nn_rng_init(&rng, config->seed);
    parameters = 0;
    stride = 0;
    for (i = 0; i < model->count; ++i) {
        nn_layer *layer = model->layers + i;
        const cmlp_activation activation = i + 1 == model->count
            ? config->output_activation : config->hidden_activation;
        float limit;
        size_t k;
        layer->in = (size_t)config->layers[i];
        layer->out = (size_t)config->layers[i + 1];
        layer->weight = parameters;
        layer->bias = parameters + layer->in * layer->out;
        parameters = layer->bias + layer->out;
        layer->input = stride;
        stride += layer->in;
        layer->output = stride;
        limit = activation == CMLP_RELU ? sqrtf(6.0f / (float)layer->in)
            : sqrtf(6.0f / ((float)layer->in + (float)layer->out));
        for (k = layer->weight; k < layer->bias; ++k) {
            const float uniform = (float)(nn_rng_next(&rng) >> 8) * (1.0f / 16777216.0f);
            model->params[k] = (2.0f * uniform - 1.0f) * limit;
        }
    }
    status = ensure_capacity(model, 1);
    if (status != CMLP_OK) { cmlp_destroy(model); return status; }
    *out = model;
    return CMLP_OK;
}

void cmlp_destroy(cmlp_model *model)
{
    if (model == NULL) return;
    free(model->widths); free(model->layers);
    free(model->params); free(model->grads); free(model->moment1); free(model->moment2);
    free(model->next_params); free(model->next_moment1); free(model->next_moment2);
    free(model->act); free(model->delta); free(model->logits);
    free(model);
}

size_t cmlp_input_size(const cmlp_model *model)
{
    return model != NULL ? (size_t)model->widths[0] : 0;
}

size_t cmlp_output_size(const cmlp_model *model)
{
    return model != NULL ? (size_t)model->widths[model->count] : 0;
}

size_t cmlp_parameter_count(const cmlp_model *model)
{
    return model != NULL ? model->parameters : 0;
}

static cmlp_status forward_one(cmlp_model *model)
{
    size_t i, o;
    for (i = 0; i < model->count; ++i) {
        const nn_layer *layer = model->layers + i;
        const cmlp_activation activation = i + 1 == model->count
            ? model->config.output_activation : model->config.hidden_activation;
        const float *input = model->act + layer->input;
        const float *weights = model->params + layer->weight;
        const float *bias = model->params + layer->bias;
        float *a = model->act + layer->output;
        for (o = 0; o < layer->out; ++o) {
            const float z = dot_product(weights + o * layer->in, input, layer->in) + bias[o];
            if (!isfinite(z)) return CMLP_NUMERIC_ERROR;
            a[o] = z;
        }
        if (activation == CMLP_SOFTMAX) {
            float maximum = a[0], denominator = 0.0f;
            memcpy(model->logits, a, layer->out * sizeof(float));
            for (o = 1; o < layer->out; ++o) if (a[o] > maximum) maximum = a[o];
            for (o = 0; o < layer->out; ++o) {
                a[o] = expf(a[o] - maximum);
                denominator += a[o];
            }
            model->softmax_max = maximum;
            model->softmax_sum = denominator;
            for (o = 0; o < layer->out; ++o) a[o] /= denominator;
        } else {
            for (o = 0; o < layer->out; ++o) a[o] = nn_activate(activation, a[o]);
        }
    }
    return CMLP_OK;
}

cmlp_status cmlp_forward(cmlp_model *model, const float *input, size_t samples, float *output)
{
    size_t i, s, input_count, total;
    cmlp_status status;
    if (model == NULL) return CMLP_INVALID_ARGUMENT;
    model->batch = 0;
    if (input == NULL || samples == 0
        || !checked_mul(samples, model->stride, &total) || !float_count(total)
        || !checked_mul(samples, cmlp_input_size(model), &input_count)
        || !float_count(input_count)) return CMLP_INVALID_ARGUMENT;
    if (!all_finite(input, input_count)) return CMLP_NUMERIC_ERROR;
    status = ensure_capacity(model, samples);
    if (status != CMLP_OK) return status;
    for (s = 0; s < samples; ++s)
        memcpy(model->act + s * model->stride, input + s * cmlp_input_size(model),
            cmlp_input_size(model) * sizeof(float));
    if (samples == 1) {
        status = forward_one(model);
        if (status != CMLP_OK) return status;
        goto copy_output;
    }
    for (i = 0; i < model->count; ++i) {
        const nn_layer *layer = model->layers + i;
        const int last = i + 1 == model->count;
        const cmlp_activation activation = last ? model->config.output_activation
            : model->config.hidden_activation;
        size_t o;
        for (o = 0; o < layer->out; ++o) {
            const float *row = model->params + layer->weight + o * layer->in;
            const float bias = model->params[layer->bias + o];
            for (s = 0; s < samples; ++s) {
                const size_t base = s * model->stride;
                const float z = dot_product(row, model->act + base + layer->input, layer->in) + bias;
                if (!isfinite(z)) return CMLP_NUMERIC_ERROR;
                model->act[base + layer->output + o] = nn_activate(activation, z);
                if (last) model->logits[s * layer->out + o] = z;
            }
        }
        if (activation == CMLP_SOFTMAX) {
            for (s = 0; s < samples; ++s) {
                float *a = model->act + s * model->stride + layer->output;
                float maximum = a[0], denominator = 0.0f;
                for (o = 1; o < layer->out; ++o) if (a[o] > maximum) maximum = a[o];
                for (o = 0; o < layer->out; ++o) {
                    a[o] = expf(a[o] - maximum);
                    denominator += a[o];
                }
                for (o = 0; o < layer->out; ++o) a[o] /= denominator;
            }
        }
    }
copy_output:
    if (output != NULL) {
        const nn_layer *last = model->layers + model->count - 1;
        for (s = 0; s < samples; ++s)
            memcpy(output + s * last->out, model->act + s * model->stride + last->output,
                last->out * sizeof(float));
    }
    model->batch = samples;
    return CMLP_OK;
}

cmlp_status cmlp_zero_grad(cmlp_model *model)
{
    if (model == NULL) return CMLP_INVALID_ARGUMENT;
    memset(model->grads, 0, model->parameters * sizeof(float));
    model->accumulated = 0;
    model->gradient_bound = 0.0;
    return CMLP_OK;
}

static cmlp_status backward_deltas(cmlp_model *model, size_t samples)
{
    size_t i, s, o;
    if (samples == 1) {
        for (i = model->count; i-- > 0;) {
            const nn_layer *layer = model->layers + i;
            const float *input = model->act + layer->input;
            const float *delta = model->delta + layer->output;
            float *gradient = model->grads + layer->weight;
            float *bias = model->grads + layer->bias;
            const double input_bound = fmax(1.0, maximum_magnitude(input, layer->in));
            const double delta_bound = maximum_magnitude(delta, layer->out);
            /* Each parameter in this layer receives one multiply/add. This
             * bound includes both roundings and subnormal rounding. If ample
             * range remains, the identical FMA kernel needs no per-vector
             * range reduction; extreme histories retain full checking. */
            const double bound = (model->gradient_bound + delta_bound * input_bound)
                * (1.0 + 2.0 * FLT_EPSILON) + FLT_MIN;
            for (o = 0; o < layer->out; ++o) {
                if (delta[o] == 0.0f) continue;
                if (bound <= 0.5 * FLT_MAX) {
                    scaled_accumulate(gradient + o * layer->in, input, layer->in, delta[o]);
                } else if (!checked_accumulate(gradient + o * layer->in, input, layer->in, delta[o])) {
                    goto numeric_error;
                }
                bias[o] += delta[o];
                if (!isfinite(bias[o])) goto numeric_error;
            }
            model->gradient_bound = bound;
            if (i != 0) {
                const float *weights = model->params + layer->weight;
                float *previous_delta = model->delta + layer->input;
                size_t k;
                memset(previous_delta, 0, layer->in * sizeof(float));
                for (o = 0; o < layer->out; ++o)
                    scaled_accumulate(previous_delta, weights + o * layer->in, layer->in, delta[o]);
                for (k = 0; k < layer->in; ++k)
                    previous_delta[k] *= nn_activation_derivative(model->config.hidden_activation, input[k]);
                if (!all_finite(previous_delta, layer->in)) goto numeric_error;
            }
        }
        model->accumulated++;
        model->batch = 0;
        return CMLP_OK;
    }
    for (i = model->count; i-- > 0;) {
        const nn_layer *layer = model->layers + i;
        for (o = 0; o < layer->out; ++o) {
            float *gradient = model->grads + layer->weight + o * layer->in;
            float bias_gradient = model->grads[layer->bias + o];
            for (s = 0; s < samples; ++s) {
                const size_t base = s * model->stride;
                const float delta = model->delta[base + layer->output + o];
                if (delta == 0.0f) continue;
                if (!checked_accumulate(gradient, model->act + base + layer->input,
                        layer->in, delta)) goto numeric_error;
                bias_gradient += delta;
            }
            if (!isfinite(bias_gradient)) goto numeric_error;
            model->grads[layer->bias + o] = bias_gradient;
        }
        if (i == 0) break;
        for (s = 0; s < samples; ++s)
            memset(model->delta + s * model->stride + layer->input, 0, layer->in * sizeof(float));
        for (o = 0; o < layer->out; ++o) {
            const float *row = model->params + layer->weight + o * layer->in;
            for (s = 0; s < samples; ++s) {
                const size_t base = s * model->stride;
                scaled_accumulate(model->delta + base + layer->input, row, layer->in,
                    model->delta[base + layer->output + o]);
            }
        }
        for (s = 0; s < samples; ++s) {
            const size_t base = s * model->stride + layer->input;
            size_t k;
            for (k = 0; k < layer->in; ++k)
                model->delta[base + k] *= nn_activation_derivative(
                    model->config.hidden_activation, model->act[base + k]);
            if (!all_finite(model->delta + base, layer->in)) goto numeric_error;
        }
    }
    model->accumulated += samples;
    model->gradient_bound = FLT_MAX;
    model->batch = 0;
    return CMLP_OK;
numeric_error:
    model->batch = 0;
    cmlp_zero_grad(model);
    return CMLP_NUMERIC_ERROR;
}

cmlp_status cmlp_backward(cmlp_model *model, const float *grad_output, size_t samples)
{
    const nn_layer *last;
    size_t s, o, total;
    if (model == NULL || grad_output == NULL || samples == 0) return CMLP_INVALID_ARGUMENT;
    if (model->batch == 0 || samples != model->batch) return CMLP_INVALID_STATE;
    if (samples > SIZE_MAX - model->accumulated) return CMLP_INVALID_STATE;
    last = model->layers + model->count - 1;
    if (!checked_mul(samples, last->out, &total) || !float_count(total))
        return CMLP_INVALID_ARGUMENT;
    if (!all_finite(grad_output, total)) return CMLP_NUMERIC_ERROR;
    for (s = 0; s < samples; ++s) {
        const float *a = model->act + s * model->stride + last->output;
        const float *g = grad_output + s * last->out;
        float *delta = model->delta + s * model->stride + last->output;
        if (model->config.output_activation == CMLP_SOFTMAX) {
            double product = 0.0;
            for (o = 0; o < last->out; ++o) product += (double)g[o] * a[o];
            for (o = 0; o < last->out; ++o) {
                const double value = (double)a[o] * ((double)g[o] - product);
                if (fabs(value) > FLT_MAX) return CMLP_NUMERIC_ERROR;
                delta[o] = (float)value;
            }
        } else {
            for (o = 0; o < last->out; ++o)
                delta[o] = g[o] * nn_activation_derivative(model->config.output_activation, a[o]);
        }
        if (!all_finite(delta, last->out)) return CMLP_NUMERIC_ERROR;
    }
    return backward_deltas(model, samples);
}

static void swap_floats(float **a, float **b)
{
    float *old = *a;
    *a = *b;
    *b = old;
}

cmlp_status cmlp_step(cmlp_model *model)
{
    size_t i = 0;
    double scale, norm = 0.0;
    if (model == NULL) return CMLP_INVALID_ARGUMENT;
    if (model->accumulated == 0 || model->steps == UINT64_MAX) return CMLP_INVALID_STATE;
    scale = 1.0 / (double)model->accumulated;
    if (model->config.gradient_clip > 0.0f) {
        for (i = 0; i < model->parameters; ++i) {
            const double g = (double)model->grads[i] * scale;
            norm += g * g;
        }
        norm = sqrt(norm);
        if (norm > model->config.gradient_clip)
            scale *= (double)model->config.gradient_clip / norm;
    }
    if (model->config.optimizer == CMLP_SGD) {
        const float rate = (float)((double)model->config.learning_rate * scale);
        const float momentum = model->config.momentum;
        i = 0;
        /* Folding the batch scale into a subnormal rate can erase a representable
         * update when the gradient is large. Keep this uncommon path in double. */
        if (rate < FLT_MIN) {
            const double precise_rate = (double)model->config.learning_rate * scale;
            for (; i < model->parameters; ++i) {
                const double velocity = (double)momentum * model->moment1[i]
                    - precise_rate * model->grads[i];
                const double parameter = (double)model->params[i] + velocity;
                if (fabs(velocity) > FLT_MAX || fabs(parameter) > FLT_MAX)
                    return CMLP_NUMERIC_ERROR;
                model->next_moment1[i] = (float)velocity;
                model->next_params[i] = (float)parameter;
            }
        }
#if defined(__AVX2__)
        {
            const __m256 m = _mm256_set1_ps(momentum), r = _mm256_set1_ps(-rate);
            for (; model->parameters - i >= 8; i += 8) {
                const __m256 velocity = multiply_add(r, _mm256_loadu_ps(model->grads + i),
                    _mm256_mul_ps(m, _mm256_loadu_ps(model->moment1 + i)));
                _mm256_storeu_ps(model->next_moment1 + i, velocity);
                _mm256_storeu_ps(model->next_params + i,
                    _mm256_add_ps(_mm256_loadu_ps(model->params + i), velocity));
            }
        }
#endif
        for (; i < model->parameters; ++i) {
            model->next_moment1[i] = momentum * model->moment1[i] - rate * model->grads[i];
            model->next_params[i] = model->params[i] + model->next_moment1[i];
        }
    } else {
        const double beta1 = model->config.beta1, beta2 = model->config.beta2;
        const double bc1 = 1.0 - pow(beta1, (double)(model->steps + 1));
        const double bc2 = 1.0 - pow(beta2, (double)(model->steps + 1));
        for (i = 0; i < model->parameters; ++i) {
            const double g = (double)model->grads[i] * scale;
            const double m = beta1 * model->moment1[i] + (1.0 - beta1) * g;
            const double v = beta2 * model->moment2[i] + (1.0 - beta2) * g * g;
            const double p = (double)model->params[i] - model->config.learning_rate
                * (m / bc1) / (sqrt(v / bc2) + model->config.epsilon);
            if (!isfinite(m) || !isfinite(v) || !isfinite(p)
                || fabs(m) > FLT_MAX || v > FLT_MAX || fabs(p) > FLT_MAX)
                return CMLP_NUMERIC_ERROR;
            model->next_moment1[i] = (float)m;
            model->next_moment2[i] = (float)v;
            model->next_params[i] = (float)p;
        }
        if (!all_finite(model->next_moment2, model->parameters)) return CMLP_NUMERIC_ERROR;
    }
    if (!all_finite(model->next_params, model->parameters)
        || !all_finite(model->next_moment1, model->parameters)) return CMLP_NUMERIC_ERROR;
    swap_floats(&model->params, &model->next_params);
    swap_floats(&model->moment1, &model->next_moment1);
    if (model->config.optimizer == CMLP_ADAM) swap_floats(&model->moment2, &model->next_moment2);
    model->steps++;
    model->batch = 0;
    return cmlp_zero_grad(model);
}

static cmlp_status classify_sample(cmlp_model *model, const float *input,
    const float *target, cmlp_metrics *metrics, int training)
{
    cmlp_status status;
    size_t o, label = 0, predicted = 0;
    const nn_layer *last;
    double sum = 0.0, loss = 0.0, target_logit = 0.0;
    float maximum;
    const float *a;
    cmlp_metrics result;
    if (model == NULL || input == NULL || target == NULL) return CMLP_INVALID_ARGUMENT;
    if (model->config.output_activation != CMLP_SOFTMAX) return CMLP_INVALID_STATE;
    if (training && model->accumulated == SIZE_MAX) return CMLP_INVALID_STATE;
    last = model->layers + model->count - 1;
    for (o = 0; o < last->out; ++o) {
        if (!isfinite(target[o])) return CMLP_NUMERIC_ERROR;
        if (target[o] < 0.0f || target[o] > 1.0f) return CMLP_INVALID_ARGUMENT;
        sum += target[o];
        if (target[o] > target[label]) label = o;
    }
    if (fabs(sum - 1.0) > 1e-5) return CMLP_INVALID_ARGUMENT;
    status = cmlp_forward(model, input, 1, NULL);
    if (status != CMLP_OK) return status;
    a = model->act + last->output;
    maximum = model->softmax_max;
    for (o = 0; o < last->out; ++o) {
        if (a[o] > a[predicted]) predicted = o;
    }
    for (o = 0; o < last->out; ++o) {
        const double shifted = (double)model->logits[o] - maximum;
        target_logit += target[o] * shifted;
        if (training)
            model->delta[last->output + o] = (float)(sum * a[o] - target[o]);
    }
    loss = sum * log((double)model->softmax_sum) - target_logit;
    if (!isfinite(loss) || loss > FLT_MAX) { model->batch = 0; return CMLP_NUMERIC_ERROR; }
    result.loss = (float)loss;
    result.correct = label == predicted ? 1u : 0u;
    if (training) {
        status = backward_deltas(model, 1);
        if (status != CMLP_OK) return status;
    }
    if (metrics != NULL) *metrics = result;
    return CMLP_OK;
}

cmlp_status cmlp_train_sample(cmlp_model *model, const float *input,
    const float *target, cmlp_metrics *metrics)
{
    return classify_sample(model, input, target, metrics, 1);
}

cmlp_status cmlp_evaluate_sample(cmlp_model *model, const float *input,
    const float *target, cmlp_metrics *metrics)
{
    return classify_sample(model, input, target, metrics, 0);
}

cmlp_status cmlp_set_learning_rate(cmlp_model *model, float rate)
{
    if (model == NULL || !isfinite(rate) || rate <= 0.0f) return CMLP_INVALID_ARGUMENT;
    model->config.learning_rate = rate;
    return CMLP_OK;
}

cmlp_status cmlp_scale_learning_rate(cmlp_model *model, float factor)
{
    float rate;
    if (model == NULL || !isfinite(factor) || factor <= 0.0f) return CMLP_INVALID_ARGUMENT;
    rate = model->config.learning_rate * factor;
    if (!isfinite(rate) || rate <= 0.0f) return CMLP_NUMERIC_ERROR;
    return cmlp_set_learning_rate(model, rate);
}

float cmlp_learning_rate(const cmlp_model *model)
{
    return model != NULL ? model->config.learning_rate : 0.0f;
}

cmlp_status cmlp_copy_from(cmlp_model *destination, const cmlp_model *source)
{
    if (destination == NULL || source == NULL) return CMLP_INVALID_ARGUMENT;
    if (destination->config.layer_count != source->config.layer_count
        || destination->config.hidden_activation != source->config.hidden_activation
        || destination->config.output_activation != source->config.output_activation
        || memcmp(destination->widths, source->widths,
            source->config.layer_count * sizeof(int)) != 0) return CMLP_INVALID_ARGUMENT;
    if (destination != source)
        memcpy(destination->params, source->params, source->parameters * sizeof(float));
    memset(destination->moment1, 0, destination->parameters * sizeof(float));
    if (destination->moment2 != NULL)
        memset(destination->moment2, 0, destination->parameters * sizeof(float));
    destination->steps = 0;
    destination->batch = 0;
    return cmlp_zero_grad(destination);
}

/* V1 is little-endian IEEE binary32, never a struct dump:
 * magic[8], version/u32, hidden/u32, output/u32, optimizer/u32, layers/u32,
 * seed/u32, rate/momentum/beta1/beta2/epsilon/clip (six f32), parameters/u64,
 * steps/u64, accumulated/u64, widths[layers]/u32, then parameter, gradient,
 * first-moment and (Adam only) second-moment arrays, followed by CRC32/u32.
 * Activations and transient scratch are intentionally not persisted. */
typedef struct nn_checkpoint {
    FILE *file;
    uint32_t crc;
    cmlp_status status;
} nn_checkpoint;

static uint32_t crc_update(uint32_t crc, const unsigned char *bytes, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i) {
        unsigned bit;
        crc ^= bytes[i];
        for (bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (uint32_t)-(int32_t)(crc & 1u));
    }
    return crc;
}

static void store_u32(unsigned char *bytes, uint32_t value)
{
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8);
    bytes[2] = (unsigned char)(value >> 16);
    bytes[3] = (unsigned char)(value >> 24);
}

static uint32_t decode_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static void checkpoint_write(nn_checkpoint *stream, const void *data, size_t count)
{
    if (stream->status != CMLP_OK) return;
    if (fwrite(data, 1, count, stream->file) != count) {
        stream->status = CMLP_IO_ERROR;
        return;
    }
    stream->crc = crc_update(stream->crc, (const unsigned char *)data, count);
}

static void checkpoint_read(nn_checkpoint *stream, void *data, size_t count)
{
    if (stream->status != CMLP_OK) return;
    if (fread(data, 1, count, stream->file) != count) {
        stream->status = ferror(stream->file) ? CMLP_IO_ERROR : CMLP_INVALID_FORMAT;
        return;
    }
    stream->crc = crc_update(stream->crc, (const unsigned char *)data, count);
}

static void write_u32(nn_checkpoint *stream, uint32_t value)
{
    unsigned char bytes[4];
    store_u32(bytes, value);
    checkpoint_write(stream, bytes, sizeof(bytes));
}

static uint32_t read_u32(nn_checkpoint *stream)
{
    unsigned char bytes[4] = {0};
    checkpoint_read(stream, bytes, sizeof(bytes));
    return decode_u32(bytes);
}

static void write_u64(nn_checkpoint *stream, uint64_t value)
{
    write_u32(stream, (uint32_t)value);
    write_u32(stream, (uint32_t)(value >> 32));
}

static uint64_t read_u64(nn_checkpoint *stream)
{
    const uint64_t low = read_u32(stream);
    const uint64_t high = read_u32(stream);
    return low | (high << 32);
}

static void write_float(nn_checkpoint *stream, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    write_u32(stream, bits);
}

static float read_float(nn_checkpoint *stream)
{
    const uint32_t bits = read_u32(stream);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void write_floats(nn_checkpoint *stream, const float *values, size_t count)
{
    unsigned char buffer[2048];
    size_t start;
    for (start = 0; start < count && stream->status == CMLP_OK;) {
        const size_t chunk = count - start < 512 ? count - start : 512;
        size_t i;
        for (i = 0; i < chunk; ++i) {
            uint32_t bits;
            memcpy(&bits, values + start + i, sizeof(bits));
            store_u32(buffer + 4 * i, bits);
        }
        checkpoint_write(stream, buffer, 4 * chunk);
        start += chunk;
    }
}

static void read_floats(nn_checkpoint *stream, float *values, size_t count)
{
    unsigned char buffer[2048];
    size_t start;
    for (start = 0; start < count && stream->status == CMLP_OK;) {
        const size_t chunk = count - start < 512 ? count - start : 512;
        size_t i;
        checkpoint_read(stream, buffer, 4 * chunk);
        if (stream->status != CMLP_OK) return;
        for (i = 0; i < chunk; ++i) {
            const uint32_t bits = decode_u32(buffer + 4 * i);
            memcpy(values + start + i, &bits, sizeof(bits));
        }
        start += chunk;
    }
}

static int binary32_supported(void)
{
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128;
}

cmlp_status cmlp_save(const cmlp_model *model, const char *path)
{
    static const unsigned char magic[8] = {'C', 'M', 'L', 'P', 'C', 'K', 'P', 'T'};
    nn_checkpoint stream;
    unsigned char checksum[4];
    size_t i;
    if (model == NULL || path == NULL || path[0] == '\0') return CMLP_INVALID_ARGUMENT;
    if (!binary32_supported()) return CMLP_INVALID_STATE;
    if (!all_finite(model->params, model->parameters)
        || !all_finite(model->grads, model->parameters)
        || !all_finite(model->moment1, model->parameters)
        || (model->moment2 != NULL && !all_finite(model->moment2, model->parameters)))
        return CMLP_NUMERIC_ERROR;
    stream.file = fopen(path, "wb");
    if (stream.file == NULL) return CMLP_IO_ERROR;
    stream.crc = UINT32_MAX;
    stream.status = CMLP_OK;
    checkpoint_write(&stream, magic, sizeof(magic));
    write_u32(&stream, 1);
    write_u32(&stream, (uint32_t)model->config.hidden_activation);
    write_u32(&stream, (uint32_t)model->config.output_activation);
    write_u32(&stream, (uint32_t)model->config.optimizer);
    write_u32(&stream, (uint32_t)model->config.layer_count);
    write_u32(&stream, model->config.seed);
    write_float(&stream, model->config.learning_rate);
    write_float(&stream, model->config.momentum);
    write_float(&stream, model->config.beta1);
    write_float(&stream, model->config.beta2);
    write_float(&stream, model->config.epsilon);
    write_float(&stream, model->config.gradient_clip);
    write_u64(&stream, (uint64_t)model->parameters);
    write_u64(&stream, model->steps);
    write_u64(&stream, (uint64_t)model->accumulated);
    for (i = 0; i < model->config.layer_count; ++i) write_u32(&stream, (uint32_t)model->widths[i]);
    write_floats(&stream, model->params, model->parameters);
    write_floats(&stream, model->grads, model->parameters);
    write_floats(&stream, model->moment1, model->parameters);
    if (model->moment2 != NULL) write_floats(&stream, model->moment2, model->parameters);
    store_u32(checksum, ~stream.crc);
    if (stream.status == CMLP_OK && fwrite(checksum, 1, sizeof(checksum), stream.file) != sizeof(checksum))
        stream.status = CMLP_IO_ERROR;
    if (fclose(stream.file) != 0) stream.status = CMLP_IO_ERROR;
    return stream.status;
}

static int checkpoint_size(FILE *file, uint64_t *size)
{
#if defined(_WIN32)
    __int64 position;
    if (_fseeki64(file, 0, SEEK_END) != 0) return 0;
    position = _ftelli64(file);
    if (position < 0 || _fseeki64(file, 0, SEEK_SET) != 0) return 0;
#else
    long position;
    if (fseek(file, 0, SEEK_END) != 0) return 0;
    position = ftell(file);
    if (position < 0 || fseek(file, 0, SEEK_SET) != 0) return 0;
#endif
    *size = (uint64_t)position;
    return 1;
}

cmlp_status cmlp_load(const char *path, cmlp_model **out)
{
    static const unsigned char expected_magic[8] = {'C', 'M', 'L', 'P', 'C', 'K', 'P', 'T'};
    unsigned char magic[8], checksum[4];
    nn_checkpoint stream;
    cmlp_model *model = NULL;
    cmlp_config config = cmlp_default_config();
    uint64_t size, expected_size, parameters, steps, accumulated;
    uint32_t version, hidden, output, optimizer, layers;
    int *widths = NULL;
    size_t i;
    if (path == NULL || path[0] == '\0' || out == NULL) return CMLP_INVALID_ARGUMENT;
    if (!binary32_supported()) return CMLP_INVALID_STATE;
    stream.file = fopen(path, "rb");
    if (stream.file == NULL) return CMLP_IO_ERROR;
    stream.status = CMLP_OK;
    stream.crc = UINT32_MAX;
    if (!checkpoint_size(stream.file, &size)) { stream.status = CMLP_IO_ERROR; goto done; }
    if (size < 92) { stream.status = CMLP_INVALID_FORMAT; goto done; }
    checkpoint_read(&stream, magic, sizeof(magic));
    version = read_u32(&stream);
    hidden = read_u32(&stream);
    output = read_u32(&stream);
    optimizer = read_u32(&stream);
    layers = read_u32(&stream);
    config.seed = read_u32(&stream);
    config.learning_rate = read_float(&stream);
    config.momentum = read_float(&stream);
    config.beta1 = read_float(&stream);
    config.beta2 = read_float(&stream);
    config.epsilon = read_float(&stream);
    config.gradient_clip = read_float(&stream);
    parameters = read_u64(&stream);
    steps = read_u64(&stream);
    accumulated = read_u64(&stream);
    if (stream.status != CMLP_OK) goto done;
    if (memcmp(magic, expected_magic, sizeof(magic)) != 0 || version != 1
        || hidden > (uint32_t)CMLP_TANH || output > (uint32_t)CMLP_SOFTMAX
        || optimizer > (uint32_t)CMLP_ADAM || layers < 2 || parameters == 0
        || accumulated > SIZE_MAX || parameters > SIZE_MAX / sizeof(float)
        || parameters > PTRDIFF_MAX / sizeof(float)
        || (uint64_t)layers * sizeof(nn_layer) > SIZE_MAX
        || (uint64_t)layers * sizeof(nn_layer) > PTRDIFF_MAX) {
        stream.status = CMLP_INVALID_FORMAT; goto done;
    }
    expected_size = UINT64_C(84) + (uint64_t)layers * 4;
    if (parameters > (UINT64_MAX - expected_size) / (optimizer == CMLP_ADAM ? 16u : 12u)
        || expected_size + parameters * (optimizer == CMLP_ADAM ? 16u : 12u) != size) {
        stream.status = CMLP_INVALID_FORMAT; goto done;
    }
    config.hidden_activation = (cmlp_activation)hidden;
    config.output_activation = (cmlp_activation)output;
    config.optimizer = (cmlp_optimizer)optimizer;
    config.layer_count = layers;
    widths = (int *)malloc((size_t)layers * sizeof(int));
    if (widths == NULL) { stream.status = CMLP_OUT_OF_MEMORY; goto done; }
    config.layers = widths;
    for (i = 0; i < layers; ++i) {
        const uint32_t width = read_u32(&stream);
        if (width == 0 || width > INT_MAX) { stream.status = CMLP_INVALID_FORMAT; goto done; }
        widths[i] = (int)width;
    }
    if (stream.status != CMLP_OK) goto done;
    {
        uint64_t actual = 0;
        for (i = 1; i < layers; ++i) {
            const uint64_t connection = ((uint64_t)widths[i - 1] + 1u) * (uint64_t)widths[i];
            if (connection > parameters - actual) { stream.status = CMLP_INVALID_FORMAT; goto done; }
            actual += connection;
        }
        if (actual != parameters || !valid_config(&config)) {
            stream.status = CMLP_INVALID_FORMAT; goto done;
        }
    }
    /* Loading does not need entropy, even when the checkpoint records seed zero. */
    {
        const uint32_t seed = config.seed;
        config.seed = 1;
        stream.status = cmlp_create(&config, &model);
        if (stream.status != CMLP_OK) goto done;
        model->config.seed = seed;
    }
    read_floats(&stream, model->params, model->parameters);
    read_floats(&stream, model->grads, model->parameters);
    read_floats(&stream, model->moment1, model->parameters);
    if (model->moment2 != NULL) read_floats(&stream, model->moment2, model->parameters);
    if (stream.status != CMLP_OK) goto done;
    if (fread(checksum, 1, sizeof(checksum), stream.file) != sizeof(checksum)) {
        stream.status = ferror(stream.file) ? CMLP_IO_ERROR : CMLP_INVALID_FORMAT;
        goto done;
    }
    if (decode_u32(checksum) != ~stream.crc || fgetc(stream.file) != EOF) {
        stream.status = CMLP_INVALID_FORMAT; goto done;
    }
    if (ferror(stream.file)) { stream.status = CMLP_IO_ERROR; goto done; }
    if (!all_finite(model->params, model->parameters) || !all_finite(model->grads, model->parameters)
        || !all_finite(model->moment1, model->parameters)
        || (model->moment2 != NULL && !all_finite(model->moment2, model->parameters))) {
        stream.status = CMLP_INVALID_FORMAT; goto done;
    }
    for (i = 0; i < model->parameters; ++i) {
        if ((accumulated == 0 && model->grads[i] != 0.0f)
            || (model->moment2 != NULL && model->moment2[i] < 0.0f)
            || (steps == 0 && (model->moment1[i] != 0.0f
                || (model->moment2 != NULL && model->moment2[i] != 0.0f)))) {
            stream.status = CMLP_INVALID_FORMAT; goto done;
        }
    }
    model->steps = steps;
    model->accumulated = (size_t)accumulated;
    model->gradient_bound = maximum_magnitude(model->grads, model->parameters);
done:
    if (fclose(stream.file) != 0 && stream.status == CMLP_OK) stream.status = CMLP_IO_ERROR;
    free(widths);
    if (stream.status != CMLP_OK) cmlp_destroy(model);
    else *out = model;
    return stream.status;
}
