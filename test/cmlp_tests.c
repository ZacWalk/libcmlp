#include "cmlp.h"
#include "../src/random.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long checks;
static const char *checkpoint_path = "cmlp-test-checkpoint.bin";
static const char *fixture_path = "cmlp-test-fixture.bin";

#define CHECK(expression) do { \
    ++checks; \
    if (!(expression)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expression); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#define OK(expression) CHECK((expression) == CMLP_OK)

static void near_value(double actual, double expected, double tolerance)
{
    ++checks;
    if (!isfinite(actual) || !isfinite(expected) || fabs(actual - expected) > tolerance) {
        fprintf(stderr, "numeric check failed: actual %.12g expected %.12g tolerance %.5g\n",
            actual, expected, tolerance);
        exit(EXIT_FAILURE);
    }
}

typedef struct snapshot {
    unsigned char *bytes;
    size_t size, count, offset;
} snapshot;

static uint32_t get_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u32(unsigned char *p, uint32_t value)
{
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

static float get_float(const snapshot *state, size_t array, size_t index)
{
    const uint32_t bits = get_u32(state->bytes + state->offset + (array * state->count + index) * 4);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void put_float(snapshot *state, size_t array, size_t index, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    put_u32(state->bytes + state->offset + (array * state->count + index) * 4, bits);
}

/* Checkpoints provide parameter fixtures without exporting mutable model internals.
 * These helpers also independently verify the versioned wire format and CRC. */
static uint32_t checksum(const unsigned char *bytes, size_t count)
{
    uint32_t crc = UINT32_MAX;
    size_t i;
    for (i = 0; i < count; ++i) {
        unsigned k;
        crc ^= bytes[i];
        for (k = 0; k < 8; ++k)
            crc = (crc & 1u) ? (crc >> 1) ^ UINT32_C(0xedb88320) : crc >> 1;
    }
    return ~crc;
}

static void write_fixture(snapshot *state, size_t length, int reseal)
{
    FILE *file;
    if (reseal) put_u32(state->bytes + state->size - 4, checksum(state->bytes, state->size - 4));
    file = fopen(fixture_path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(state->bytes, 1, length, file) == length);
    CHECK(fclose(file) == 0);
}

static snapshot capture(const cmlp_model *model)
{
    snapshot state;
    long length;
    FILE *file;
    OK(cmlp_save(model, checkpoint_path));
    file = fopen(checkpoint_path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    length = ftell(file);
    CHECK(length >= 92);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    state.size = (size_t)length;
    state.bytes = (unsigned char *)malloc(state.size);
    CHECK(state.bytes != NULL);
    CHECK(fread(state.bytes, 1, state.size, file) == state.size);
    CHECK(fclose(file) == 0);
    CHECK(remove(checkpoint_path) == 0);
    CHECK(memcmp(state.bytes, "CMLPCKPT", 8) == 0);
    CHECK(get_u32(state.bytes + 8) == 1);
    CHECK(get_u32(state.bytes + 60) == 0);
    state.count = get_u32(state.bytes + 56);
    state.offset = 80 + (size_t)get_u32(state.bytes + 24) * 4;
    CHECK(checksum(state.bytes, state.size - 4) == get_u32(state.bytes + state.size - 4));
    CHECK(state.count == cmlp_parameter_count(model));
    return state;
}

static cmlp_model *restore(snapshot *state)
{
    cmlp_model *model = NULL;
    write_fixture(state, state->size, 1);
    OK(cmlp_load(fixture_path, &model));
    CHECK(remove(fixture_path) == 0);
    return model;
}

static cmlp_model *make_model(const int *widths, size_t count,
    cmlp_activation hidden, cmlp_activation output, cmlp_optimizer optimizer)
{
    cmlp_config config = cmlp_default_config();
    cmlp_model *model = NULL;
    config.layers = widths;
    config.layer_count = count;
    config.hidden_activation = hidden;
    config.output_activation = output;
    config.optimizer = optimizer;
    config.seed = 12345;
    config.learning_rate = 0.025f;
    config.momentum = 0.6f;
    OK(cmlp_create(&config, &model));
    return model;
}

static void compare_parameters(const cmlp_model *a, const cmlp_model *b, double tolerance)
{
    snapshot sa = capture(a), sb = capture(b);
    size_t i;
    CHECK(sa.count == sb.count);
    for (i = 0; i < sa.count; ++i) near_value(get_float(&sa, 0, i), get_float(&sb, 0, i), tolerance);
    free(sa.bytes); free(sb.bytes);
}

static double squared_loss(cmlp_model *model, const float *input, const float *target, size_t samples)
{
    float output[16];
    double loss = 0.0;
    size_t i, count = cmlp_output_size(model) * samples;
    CHECK(count <= 16);
    OK(cmlp_forward(model, input, samples, output));
    for (i = 0; i < count; ++i) {
        const double difference = (double)output[i] - target[i];
        loss += 0.5 * difference * difference;
    }
    return loss / (double)samples;
}

static void test_activation_gradients(void)
{
    const int widths[] = {3, 4, 2};
    const float input[] = {0.2f, -0.3f, 0.7f, -0.4f, 0.5f, 0.1f};
    const float target[] = {0.12f, -0.14f, 0.2f, 0.3f};
    int hidden, output_kind;
    for (hidden = CMLP_LINEAR; hidden <= CMLP_TANH; ++hidden) {
        for (output_kind = CMLP_LINEAR; output_kind <= CMLP_SOFTMAX; ++output_kind) {
            cmlp_model *model = make_model(widths, 3, (cmlp_activation)hidden,
                (cmlp_activation)output_kind, CMLP_SGD);
            snapshot initial = capture(model), gradients;
            float output[4], derivatives[4];
            size_t k;
            for (k = 0; k < initial.count; ++k)
                put_float(&initial, 0, k, 0.04f * (float)((int)(k % 7) - 3));
            put_float(&initial, 0, 12, 0.7f);
            put_float(&initial, 0, 13, -0.6f);
            put_float(&initial, 0, 14, 0.4f);
            put_float(&initial, 0, 15, -0.5f);
            put_float(&initial, 0, 24, 0.35f);
            put_float(&initial, 0, 25, 0.45f);
            cmlp_destroy(model);
            model = restore(&initial);
            OK(cmlp_forward(model, input, 2, output));
            for (k = 0; k < 4; ++k) derivatives[k] = output[k] - target[k];
            OK(cmlp_backward(model, derivatives, 2));
            gradients = capture(model);
            for (k = 0; k < initial.count; ++k) {
                const float original = get_float(&initial, 0, k), epsilon = 0.001f;
                double plus, minus, estimate;
                cmlp_model *perturbed;
                put_float(&initial, 0, k, original + epsilon);
                perturbed = restore(&initial);
                plus = squared_loss(perturbed, input, target, 2);
                cmlp_destroy(perturbed);
                put_float(&initial, 0, k, original - epsilon);
                perturbed = restore(&initial);
                minus = squared_loss(perturbed, input, target, 2);
                cmlp_destroy(perturbed);
                put_float(&initial, 0, k, original);
                estimate = (plus - minus) / (2.0 * epsilon);
                near_value(get_float(&gradients, 1, k) / 2.0, estimate, 8e-5);
            }
            free(initial.bytes); free(gradients.bytes);
            cmlp_destroy(model);
        }
    }
}

static void test_cross_entropy(void)
{
    const int widths[] = {3, 4, 3};
    const float input[] = {0.4f, -0.2f, 0.7f};
    const float target[] = {0.2f, 0.3f, 0.5f};
    int hidden;
    for (hidden = CMLP_LINEAR; hidden <= CMLP_TANH; ++hidden) {
        cmlp_model *model = make_model(widths, 3, (cmlp_activation)hidden, CMLP_SOFTMAX, CMLP_SGD);
        cmlp_model *generic = make_model(widths, 3, (cmlp_activation)hidden, CMLP_SOFTMAX, CMLP_SGD);
        snapshot initial = capture(model), gradient;
        float output[3], derivative[3];
        cmlp_metrics metrics;
        size_t k;
        OK(cmlp_train_sample(model, input, target, &metrics));
        CHECK(isfinite(metrics.loss) && metrics.loss > 0.0f);
        gradient = capture(model);
        OK(cmlp_forward(generic, input, 1, output));
        for (k = 0; k < 3; ++k) derivative[k] = -target[k] / output[k];
        OK(cmlp_backward(generic, derivative, 1));
        OK(cmlp_step(generic));
        for (k = 0; k < initial.count; ++k) {
            cmlp_model *perturbed;
            float plus, minus;
            const float original = get_float(&initial, 0, k), epsilon = 0.001f;
            put_float(&initial, 0, k, original + epsilon);
            perturbed = restore(&initial);
            OK(cmlp_evaluate_sample(perturbed, input, target, &metrics));
            plus = metrics.loss;
            cmlp_destroy(perturbed);
            put_float(&initial, 0, k, original - epsilon);
            perturbed = restore(&initial);
            OK(cmlp_evaluate_sample(perturbed, input, target, &metrics));
            minus = metrics.loss;
            cmlp_destroy(perturbed);
            put_float(&initial, 0, k, original);
            near_value(get_float(&gradient, 1, k), (plus - minus) / (2.0 * epsilon), 1.5e-4);
        }
        OK(cmlp_step(model));
        compare_parameters(model, generic, 1e-6);
        free(initial.bytes); free(gradient.bytes);
        cmlp_destroy(model); cmlp_destroy(generic);
    }
    {
        const int topology[] = {1, 2};
        const float x[] = {1.0f}, y[] = {0.0f, 1.0f};
        cmlp_model *model = make_model(topology, 2, CMLP_LINEAR, CMLP_SOFTMAX, CMLP_SGD);
        snapshot state = capture(model);
        cmlp_metrics metrics;
        float output[2];
        put_float(&state, 0, 0, 1000.0f);
        put_float(&state, 0, 1, -1000.0f);
        cmlp_destroy(model);
        model = restore(&state);
        OK(cmlp_evaluate_sample(model, x, y, &metrics));
        near_value(metrics.loss, 2000.0, 1e-4);
        CHECK(metrics.correct == 0);
        OK(cmlp_train_sample(model, x, y, NULL));
        OK(cmlp_step(model));
        OK(cmlp_forward(model, x, 1, output));
        near_value(output[0] + output[1], 1.0, 1e-6);
        free(state.bytes); cmlp_destroy(model);
    }
}

static void test_gradient_range_bounds(void)
{
    const int widths[] = {1, 1};
    const float x[] = {1.0f};
    cmlp_model *model;
    {
        const int wide[] = {17, 1};
        const float gradient[] = {2.0f};
        float inputs[17] = {0};
        snapshot state;
        size_t i;
        model = make_model(wide, 2, CMLP_LINEAR, CMLP_LINEAR, CMLP_SGD);
        state = capture(model);
        for (i = 0; i < state.count; ++i) put_float(&state, 0, i, 0.0f);
        cmlp_destroy(model); model = restore(&state);
        for (i = 0; i < 17; ++i) {
            inputs[i] = FLT_MAX;
            OK(cmlp_forward(model, inputs, 1, NULL));
            CHECK(cmlp_backward(model, gradient, 1) == CMLP_NUMERIC_ERROR);
            CHECK(cmlp_step(model) == CMLP_INVALID_STATE);
            inputs[i] = 0.0f;
        }
        free(state.bytes); cmlp_destroy(model);
    }
    {
        const float first[] = {FLT_MAX * 0.9f}, next[] = {FLT_MAX * 0.2f};
        const float quarter[] = {FLT_MAX * 0.25f};
        snapshot state;
        size_t i;
        model = make_model(widths, 2, CMLP_LINEAR, CMLP_LINEAR, CMLP_SGD);
        OK(cmlp_forward(model, x, 1, NULL)); OK(cmlp_backward(model, first, 1));
        state = capture(model);
        cmlp_destroy(model); model = restore(&state);
        OK(cmlp_forward(model, x, 1, NULL));
        CHECK(cmlp_backward(model, next, 1) == CMLP_NUMERIC_ERROR);
        CHECK(cmlp_step(model) == CMLP_INVALID_STATE);
        for (i = 0; i < 4; ++i) {
            OK(cmlp_forward(model, x, 1, NULL)); OK(cmlp_backward(model, quarter, 1));
        }
        OK(cmlp_forward(model, x, 1, NULL));
        CHECK(cmlp_backward(model, quarter, 1) == CMLP_NUMERIC_ERROR);
        free(state.bytes); cmlp_destroy(model);
    }
}

static void test_batching_and_retention(void)
{
    const int widths[] = {17, 9, 3};
    float input[5 * 17], output[15], expected[15], derivative[15];
    cmlp_model *batch = make_model(widths, 3, CMLP_TANH, CMLP_SOFTMAX, CMLP_SGD);
    cmlp_model *single = make_model(widths, 3, CMLP_TANH, CMLP_SOFTMAX, CMLP_SGD);
    size_t i, s;
    for (i = 0; i < 85; ++i) input[i] = (float)((int)(i % 11) - 5) * 0.12f;
    for (i = 0; i < 15; ++i) derivative[i] = (float)((int)(i % 5) - 2) * 0.17f;
    OK(cmlp_forward(batch, input, 5, output));
    for (s = 0; s < 5; ++s) {
        OK(cmlp_forward(single, input + s * 17, 1, expected + s * 3));
        OK(cmlp_backward(single, derivative + s * 3, 1));
    }
    for (i = 0; i < 15; ++i) near_value(output[i], expected[i], 1e-6);
    /* Forward owns a retained copy, not the caller's input or output buffers. */
    for (i = 0; i < 85; ++i) input[i] = NAN;
    for (i = 0; i < 15; ++i) output[i] = NAN;
    CHECK(cmlp_backward(batch, derivative, 4) == CMLP_INVALID_STATE);
    OK(cmlp_backward(batch, derivative, 5));
    CHECK(cmlp_backward(batch, derivative, 5) == CMLP_INVALID_STATE);
    OK(cmlp_step(batch));
    OK(cmlp_step(single));
    compare_parameters(batch, single, 2e-6);
    CHECK(cmlp_step(batch) == CMLP_INVALID_STATE);
    CHECK(cmlp_backward(batch, derivative, 5) == CMLP_INVALID_STATE);
    OK(cmlp_zero_grad(batch));
    for (i = 0; i < 85; ++i) input[i] = (float)(i % 3) * 0.1f;
    OK(cmlp_forward(batch, input, 2, NULL));
    OK(cmlp_backward(batch, derivative, 2));
    OK(cmlp_forward(batch, input + 34, 3, NULL));
    OK(cmlp_backward(batch, derivative + 6, 3));
    OK(cmlp_forward(single, input, 5, NULL));
    OK(cmlp_backward(single, derivative, 5));
    OK(cmlp_step(batch));
    OK(cmlp_step(single));
    compare_parameters(batch, single, 3e-6);
    cmlp_destroy(batch); cmlp_destroy(single);
}

static void test_optimizers(void)
{
    const int widths[] = {2, 2};
    const float input[] = {1.0f, -2.0f, -0.5f, 0.75f};
    const float derivatives[][4] = {
        {3.0f, -4.0f, 1.5f, 2.0f},
        {-2.0f, 1.0f, 5.0f, -1.0f},
        {0.1f, -0.2f, 0.4f, 0.7f},
        {8.0f, -3.0f, -7.0f, 4.0f}
    };
    int optimizer, clipping;
    for (optimizer = CMLP_SGD; optimizer <= CMLP_ADAM; ++optimizer) {
        for (clipping = 0; clipping < 2; ++clipping) {
            cmlp_model *model = NULL;
            cmlp_config config = cmlp_default_config();
            snapshot state;
            double parameters[6], m[6] = {0}, v[6] = {0};
            size_t i, step;
            config.layers = widths; config.layer_count = 2; config.seed = 7;
            config.hidden_activation = CMLP_LINEAR; config.output_activation = CMLP_LINEAR;
            config.optimizer = (cmlp_optimizer)optimizer;
            config.learning_rate = 0.03f; config.momentum = 0.7f;
            config.beta1 = 0.8f; config.beta2 = 0.95f; config.epsilon = 0.0001f;
            config.gradient_clip = clipping ? 0.6f : 0.0f;
            OK(cmlp_create(&config, &model));
            state = capture(model);
            for (i = 0; i < 6; ++i) parameters[i] = get_float(&state, 0, i);
            free(state.bytes);
            for (step = 0; step < 4; ++step) {
                double g[6] = {0}, norm = 0.0, factor = 1.0;
                size_t s, o, k;
                for (s = 0; s < 2; ++s) {
                    for (o = 0; o < 2; ++o) {
                        for (k = 0; k < 2; ++k)
                            g[o * 2 + k] += 0.5 * derivatives[step][s * 2 + o] * input[s * 2 + k];
                        g[4 + o] += 0.5 * derivatives[step][s * 2 + o];
                    }
                }
                for (i = 0; i < 6; ++i) norm += g[i] * g[i];
                norm = sqrt(norm);
                if (clipping && norm > config.gradient_clip) factor = config.gradient_clip / norm;
                for (i = 0; i < 6; ++i) {
                    g[i] *= factor;
                    if (optimizer == CMLP_SGD) {
                        m[i] = (float)(config.momentum * m[i] - config.learning_rate * g[i]);
                        parameters[i] = (float)(parameters[i] + m[i]);
                    } else {
                        const double beta1 = config.beta1, beta2 = config.beta2;
                        const double next_m = beta1 * m[i] + (1.0 - beta1) * g[i];
                        const double next_v = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                        parameters[i] = (float)(parameters[i] - config.learning_rate
                            * (next_m / (1.0 - pow(beta1, (double)step + 1.0)))
                            / (sqrt(next_v / (1.0 - pow(beta2, (double)step + 1.0))) + config.epsilon));
                        m[i] = (float)next_m; v[i] = (float)next_v;
                    }
                }
                OK(cmlp_forward(model, input, 2, NULL));
                OK(cmlp_backward(model, derivatives[step], 2));
                OK(cmlp_step(model));
                state = capture(model);
                CHECK(get_u32(state.bytes + 64) == step + 1);
                CHECK(get_u32(state.bytes + 72) == 0);
                for (i = 0; i < 6; ++i) {
                    near_value(get_float(&state, 0, i), parameters[i], 2e-6);
                    near_value(get_float(&state, 1, i), 0.0, 0.0);
                    near_value(get_float(&state, 2, i), m[i], 2e-6);
                    if (optimizer == CMLP_ADAM)
                        near_value(get_float(&state, 3, i), v[i], 2e-6);
                }
                free(state.bytes);
            }
            cmlp_destroy(model);
        }
    }
}

static void test_initialization_and_rng(void)
{
    const int widths[] = {100, 100, 10};
    cmlp_model *a, *b;
    cmlp_config config = cmlp_default_config();
    nn_rng rng = {0}, repeat;
    snapshot state;
    double mean = 0.0, squares = 0.0;
    size_t i;
    CHECK(nn_rng_next(&rng) == UINT32_C(0xe220a839));
    nn_rng_init(&rng, 1); nn_rng_init(&repeat, 1);
    CHECK(nn_rng_next(&rng) == UINT32_C(0x910a2dec));
    (void)nn_rng_next(&repeat);
    for (i = 0; i < 1000; ++i) CHECK(nn_rng_next(&rng) == nn_rng_next(&repeat));
    CHECK(nn_rng_bounded(&rng, 0) == 0);
    CHECK(nn_rng_bounded(&rng, 1) == 0);
    for (i = 0; i < 1000; ++i) {
        CHECK(nn_rng_bounded(&rng, 17) < 17);
        CHECK(nn_rng_bounded(&rng, UINT32_MAX) < UINT32_MAX);
    }
    CHECK(nn_random_seed() != 0);
    CHECK(config.hidden_activation == CMLP_SIGMOID && config.output_activation == CMLP_SOFTMAX);
    CHECK(config.optimizer == CMLP_SGD && config.gradient_clip == 0.0f);
    near_value(config.learning_rate, LEARNING_RATE, 0.0);
    near_value(config.momentum, MOMENTUM, 0.0);
    a = make_model(widths, 3, CMLP_RELU, CMLP_LINEAR, CMLP_SGD);
    b = make_model(widths, 3, CMLP_RELU, CMLP_LINEAR, CMLP_SGD);
    compare_parameters(a, b, 0.0);
    state = capture(a);
    for (i = 0; i < 10000; ++i) {
        const double value = get_float(&state, 0, i);
        CHECK(fabs(value) <= sqrt(6.0 / 100.0));
        mean += value; squares += value * value;
    }
    near_value(mean / 10000.0, 0.0, 0.006);
    near_value(squares / 10000.0, 2.0 / 100.0, 0.0015);
    for (i = 10000; i < 10100; ++i) near_value(get_float(&state, 0, i), 0.0, 0.0);
    for (i = 10100; i < 11100; ++i)
        CHECK(fabs(get_float(&state, 0, i)) <= sqrt(6.0 / 110.0) + 1e-7);
    for (i = 11100; i < 11110; ++i) near_value(get_float(&state, 0, i), 0.0, 0.0);
    free(state.bytes); cmlp_destroy(a); cmlp_destroy(b);
    {
        int topology[] = {2, 3, 1};
        config.layers = topology; config.layer_count = 3; config.seed = 0;
        OK(cmlp_create(&config, &a));
        topology[0] = 99;
        CHECK(cmlp_input_size(a) == 2 && cmlp_output_size(a) == 1);
        CHECK(cmlp_parameter_count(a) == 13);
        cmlp_destroy(a);
    }
}

static void test_invalid_inputs(void)
{
    const int widths[] = {2, 3, 2};
    const int bad_widths[] = {2, 0};
    const int enormous[] = {INT_MAX, INT_MAX, INT_MAX};
    const float input[] = {0.2f, 0.3f}, target[] = {0.0f, 1.0f}, derivative[] = {0.2f, -0.3f};
    float bad[] = {NAN, 1.0f}, output[2];
    cmlp_model *model = make_model(widths, 3, CMLP_SIGMOID, CMLP_SOFTMAX, CMLP_SGD);
    cmlp_model *sentinel = model;
    cmlp_config config = cmlp_default_config(), valid;
    cmlp_metrics metrics = {123.0f, 45};
    float original_rate = cmlp_learning_rate(model);
    config.layers = widths; config.layer_count = 3;
    valid = config;
    CHECK(cmlp_create(NULL, &sentinel) == CMLP_INVALID_ARGUMENT && sentinel == model);
    CHECK(cmlp_create(&config, NULL) == CMLP_INVALID_ARGUMENT);
    config.layer_count = 1;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT && sentinel == model);
    config = valid; config.layer_count = SIZE_MAX;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.layers = bad_widths; config.layer_count = 2;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.layers = enormous;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.hidden_activation = CMLP_SOFTMAX;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.output_activation = (cmlp_activation)99;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.optimizer = (cmlp_optimizer)99;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.learning_rate = INFINITY;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.learning_rate = 0.0f;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.momentum = 1.0f;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.beta1 = -0.1f;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.beta2 = NAN;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.epsilon = 0.0f;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    config = valid; config.gradient_clip = -1.0f;
    CHECK(cmlp_create(&config, &sentinel) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_forward(NULL, input, 1, output) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_forward(model, NULL, 1, output) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_forward(model, input, 0, output) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_forward(model, input, SIZE_MAX, output) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_forward(model, bad, 1, output) == CMLP_NUMERIC_ERROR);
    bad[0] = INFINITY;
    CHECK(cmlp_forward(model, bad, 1, output) == CMLP_NUMERIC_ERROR);
    CHECK(cmlp_backward(model, derivative, 1) == CMLP_INVALID_STATE);
    CHECK(cmlp_backward(NULL, derivative, 1) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_step(NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_zero_grad(NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_step(model) == CMLP_INVALID_STATE);
    OK(cmlp_forward(model, input, 1, NULL));
    CHECK(cmlp_backward(model, bad, 1) == CMLP_NUMERIC_ERROR);
    OK(cmlp_backward(model, derivative, 1));
    OK(cmlp_zero_grad(model));
    CHECK(cmlp_step(model) == CMLP_INVALID_STATE);
    OK(cmlp_forward(model, input, 1, NULL));
    CHECK(cmlp_forward(model, input, 0, NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_backward(model, derivative, 1) == CMLP_INVALID_STATE);
    CHECK(cmlp_train_sample(model, input, bad, &metrics) == CMLP_NUMERIC_ERROR);
    near_value(metrics.loss, 123.0, 0.0); CHECK(metrics.correct == 45);
    bad[0] = -0.1f;
    CHECK(cmlp_train_sample(model, input, bad, NULL) == CMLP_INVALID_ARGUMENT);
    bad[0] = 0.5f;
    CHECK(cmlp_evaluate_sample(model, input, bad, NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_set_learning_rate(model, NAN) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_set_learning_rate(model, 0.0f) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_scale_learning_rate(model, -1.0f) == CMLP_INVALID_ARGUMENT);
    near_value(cmlp_learning_rate(model), original_rate, 0.0);
    OK(cmlp_scale_learning_rate(model, 0.5f));
    near_value(cmlp_learning_rate(model), original_rate * 0.5, 1e-8);
    OK(cmlp_set_learning_rate(model, FLT_MAX));
    CHECK(cmlp_scale_learning_rate(model, 2.0f) == CMLP_NUMERIC_ERROR);
    near_value(cmlp_learning_rate(model), FLT_MAX, 0.0);
    CHECK(cmlp_input_size(NULL) == 0 && cmlp_output_size(NULL) == 0 && cmlp_parameter_count(NULL) == 0);
    CHECK(cmlp_learning_rate(NULL) == 0.0f);
    CHECK(strcmp(cmlp_status_string(CMLP_NUMERIC_ERROR), "success") != 0);
    CHECK(cmlp_status_string((cmlp_status)99) != NULL);
    CHECK(cmlp_save(NULL, checkpoint_path) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_save(model, NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_load(NULL, &sentinel) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_load(checkpoint_path, NULL) == CMLP_INVALID_ARGUMENT);
    CHECK(cmlp_load("cmlp-nonexistent-file.bin", &sentinel) == CMLP_IO_ERROR && sentinel == model);
    CHECK(cmlp_save(model, ".") == CMLP_IO_ERROR);
    cmlp_destroy(model);
    model = make_model(widths, 3, CMLP_RELU, CMLP_LINEAR, CMLP_SGD);
    CHECK(cmlp_train_sample(model, input, target, NULL) == CMLP_INVALID_STATE);
    CHECK(cmlp_evaluate_sample(model, input, target, NULL) == CMLP_INVALID_STATE);
    cmlp_destroy(model); cmlp_destroy(NULL);
}

static void test_numeric_failures(void)
{
    const int widths[] = {1, 1};
    const float x[] = {1.0f}, huge[] = {FLT_MAX};
    cmlp_model *model = make_model(widths, 2, CMLP_LINEAR, CMLP_LINEAR, CMLP_SGD);
    snapshot original = capture(model), failed;
    float output;
    put_float(&original, 0, 0, 2.0f);
    cmlp_destroy(model); model = restore(&original);
    CHECK(cmlp_forward(model, huge, 1, &output) == CMLP_NUMERIC_ERROR);
    CHECK(cmlp_backward(model, x, 1) == CMLP_INVALID_STATE);
    OK(cmlp_forward(model, x, 1, NULL));
    OK(cmlp_backward(model, huge, 1));
    OK(cmlp_set_learning_rate(model, FLT_MAX));
    CHECK(cmlp_step(model) == CMLP_NUMERIC_ERROR);
    failed = capture(model);
    near_value(get_float(&failed, 0, 0), 2.0, 0.0);
    near_value(get_float(&failed, 2, 0), 0.0, 0.0);
    CHECK(get_u32(failed.bytes + 72) == 1 && get_u32(failed.bytes + 64) == 0);
    free(failed.bytes);
    OK(cmlp_zero_grad(model));
    OK(cmlp_set_learning_rate(model, 0.01f));
    {
        const float two[] = {2.0f};
        OK(cmlp_forward(model, two, 1, NULL));
        CHECK(cmlp_backward(model, huge, 1) == CMLP_NUMERIC_ERROR);
        CHECK(cmlp_step(model) == CMLP_INVALID_STATE);
    }
    OK(cmlp_forward(model, x, 1, NULL));
    OK(cmlp_backward(model, x, 1));
    OK(cmlp_step(model));
    free(original.bytes); cmlp_destroy(model);
    {
        cmlp_config config = cmlp_default_config();
        const float inputs[] = {1.0f, 1.0f};
        const float gradients[] = {FLT_MAX / 4.0f, FLT_MAX / 4.0f};
        snapshot state;
        config.layers = widths; config.layer_count = 2; config.seed = 1;
        config.output_activation = CMLP_LINEAR;
        config.learning_rate = nextafterf(0.0f, 1.0f);
        config.momentum = 0.0f;
        OK(cmlp_create(&config, &model));
        state = capture(model);
        put_float(&state, 0, 0, 0.0f);
        cmlp_destroy(model); model = restore(&state);
        OK(cmlp_forward(model, inputs, 2, NULL));
        OK(cmlp_backward(model, gradients, 2));
        OK(cmlp_step(model));
        free(state.bytes); state = capture(model);
        near_value(get_float(&state, 0, 0), -(double)config.learning_rate * gradients[0], 1e-13);
        near_value(get_float(&state, 0, 1), -(double)config.learning_rate * gradients[0], 1e-13);
        free(state.bytes); cmlp_destroy(model);
        config.learning_rate = 0.01f; config.optimizer = CMLP_ADAM;
        OK(cmlp_create(&config, &model));
        OK(cmlp_forward(model, x, 1, NULL)); OK(cmlp_backward(model, huge, 1));
        CHECK(cmlp_step(model) == CMLP_NUMERIC_ERROR);
        state = capture(model);
        CHECK(get_u32(state.bytes + 64) == 0 && get_u32(state.bytes + 72) == 1);
        free(state.bytes); cmlp_destroy(model);
        config.gradient_clip = 10.0f;
        OK(cmlp_create(&config, &model));
        OK(cmlp_forward(model, x, 1, NULL)); OK(cmlp_backward(model, huge, 1));
        OK(cmlp_step(model));
        cmlp_destroy(model);
    }
    {
        const int wide[] = {17, 17};
        float inputs[17] = {0}, gradients[17] = {0};
        size_t i;
        model = make_model(wide, 2, CMLP_LINEAR, CMLP_LINEAR, CMLP_SGD);
        for (i = 0; i < 17; ++i) {
            inputs[i] = i % 2 == 0 ? NAN : -INFINITY;
            CHECK(cmlp_forward(model, inputs, 1, NULL) == CMLP_NUMERIC_ERROR);
            inputs[i] = 0.0f;
            OK(cmlp_forward(model, inputs, 1, NULL));
            gradients[i] = i % 2 == 0 ? INFINITY : NAN;
            CHECK(cmlp_backward(model, gradients, 1) == CMLP_NUMERIC_ERROR);
            gradients[i] = 0.0f;
        }
        OK(cmlp_backward(model, gradients, 1));
        OK(cmlp_step(model));
        cmlp_destroy(model);
    }
    {
        snapshot state;
        model = make_model(widths, 2, CMLP_LINEAR, CMLP_RELU, CMLP_SGD);
        state = capture(model); put_float(&state, 0, 0, 0.0f);
        cmlp_destroy(model); model = restore(&state);
        OK(cmlp_forward(model, x, 1, &output)); near_value(output, 0.0, 0.0);
        OK(cmlp_backward(model, x, 1));
        free(state.bytes); state = capture(model);
        near_value(get_float(&state, 1, 0), 0.0, 0.0);
        near_value(get_float(&state, 1, 1), 0.0, 0.0);
        free(state.bytes); cmlp_destroy(model);
    }
}

static void test_persistence_and_copy(void)
{
    const int widths[] = {2, 5, 2};
    const float input[] = {0.1f, 0.3f, -0.2f, 0.6f}, derivative[] = {0.5f, -0.7f, 0.3f, 0.1f};
    int optimizer;
    for (optimizer = CMLP_SGD; optimizer <= CMLP_ADAM; ++optimizer) {
        cmlp_model *model = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, (cmlp_optimizer)optimizer);
        cmlp_model *loaded, *target, *fresh;
        snapshot state;
        size_t step, i;
        float a[4], b[4];
        for (step = 0; step < 3; ++step) {
            OK(cmlp_forward(model, input, 2, NULL));
            OK(cmlp_backward(model, derivative, 2));
            OK(cmlp_step(model));
        }
        OK(cmlp_set_learning_rate(model, 0.003f));
        OK(cmlp_forward(model, input, 2, NULL));
        OK(cmlp_backward(model, derivative, 2));
        state = capture(model);
        loaded = restore(&state);
        CHECK(cmlp_backward(loaded, derivative, 2) == CMLP_INVALID_STATE);
        near_value(cmlp_learning_rate(loaded), 0.003f, 0.0);
        for (step = 0; step < 3; ++step) {
            OK(cmlp_step(model)); OK(cmlp_step(loaded));
            compare_parameters(model, loaded, 0.0);
            OK(cmlp_forward(model, input, 2, a));
            OK(cmlp_forward(loaded, input, 2, b));
            for (i = 0; i < 4; ++i) near_value(a[i], b[i], 0.0);
            OK(cmlp_backward(model, derivative, 2));
            OK(cmlp_backward(loaded, derivative, 2));
        }
        target = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, (cmlp_optimizer)optimizer);
        fresh = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, (cmlp_optimizer)optimizer);
        OK(cmlp_forward(target, input, 2, NULL));
        OK(cmlp_backward(target, derivative, 2)); OK(cmlp_step(target));
        OK(cmlp_forward(target, input, 2, NULL)); OK(cmlp_backward(target, derivative, 2));
        OK(cmlp_copy_from(target, model)); OK(cmlp_copy_from(fresh, model));
        near_value(cmlp_learning_rate(target), 0.025f, 0.0);
        CHECK(cmlp_step(target) == CMLP_INVALID_STATE);
        CHECK(cmlp_backward(target, derivative, 2) == CMLP_INVALID_STATE);
        compare_parameters(target, model, 0.0);
        for (step = 0; step < 3; ++step) {
            OK(cmlp_forward(target, input, 2, NULL)); OK(cmlp_forward(fresh, input, 2, NULL));
            OK(cmlp_backward(target, derivative, 2)); OK(cmlp_backward(fresh, derivative, 2));
            OK(cmlp_step(target)); OK(cmlp_step(fresh));
            compare_parameters(target, fresh, 0.0);
        }
        CHECK(cmlp_copy_from(NULL, model) == CMLP_INVALID_ARGUMENT);
        CHECK(cmlp_copy_from(model, NULL) == CMLP_INVALID_ARGUMENT);
        OK(cmlp_copy_from(target, target));
        CHECK(cmlp_step(target) == CMLP_INVALID_STATE);
        free(state.bytes);
        cmlp_destroy(model); cmlp_destroy(loaded); cmlp_destroy(target); cmlp_destroy(fresh);
    }
    {
        const int mismatch_widths[] = {2, 4, 2};
        cmlp_model *a = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, CMLP_SGD);
        cmlp_model *b = make_model(mismatch_widths, 3, CMLP_TANH, CMLP_LINEAR, CMLP_SGD);
        cmlp_model *c = make_model(widths, 3, CMLP_RELU, CMLP_LINEAR, CMLP_SGD);
        cmlp_model *d = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, CMLP_ADAM);
        CHECK(cmlp_copy_from(a, b) == CMLP_INVALID_ARGUMENT);
        CHECK(cmlp_copy_from(a, c) == CMLP_INVALID_ARGUMENT);
        OK(cmlp_copy_from(d, a));
        compare_parameters(a, d, 0.0);
        cmlp_destroy(a); cmlp_destroy(b); cmlp_destroy(c); cmlp_destroy(d);
    }
}

static void expect_bad_checkpoint(snapshot *state, cmlp_model *sentinel, size_t length, int reseal)
{
    cmlp_model *result = sentinel;
    write_fixture(state, length, reseal);
    CHECK(cmlp_load(fixture_path, &result) == CMLP_INVALID_FORMAT);
    CHECK(result == sentinel);
    CHECK(remove(fixture_path) == 0);
}

static void test_checkpoint_corruption(void)
{
    const int widths[] = {2, 3, 2};
    const float input[] = {0.2f, 0.1f}, gradient[] = {0.3f, -0.4f};
    cmlp_model *model = make_model(widths, 3, CMLP_TANH, CMLP_LINEAR, CMLP_ADAM);
    snapshot state = capture(model);
    unsigned char *original = (unsigned char *)malloc(state.size);
    size_t i;
    CHECK(original != NULL);
    memcpy(original, state.bytes, state.size);
    for (i = 0; i < state.size; ++i) expect_bad_checkpoint(&state, model, i, 0);
    for (i = 0; i < state.size; ++i) {
        state.bytes[i] ^= 1u;
        expect_bad_checkpoint(&state, model, state.size, 0);
        state.bytes[i] ^= 1u;
    }
    {
        const size_t fields[] = {8, 12, 16, 20, 24, 56, 80};
        for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
            put_u32(state.bytes + fields[i], UINT32_MAX);
            expect_bad_checkpoint(&state, model, state.size, 1);
            memcpy(state.bytes, original, state.size);
        }
    }
    put_u32(state.bytes + 32, UINT32_C(0x7fc00000));
    expect_bad_checkpoint(&state, model, state.size, 1);
    memcpy(state.bytes, original, state.size);
    put_float(&state, 0, 0, INFINITY);
    expect_bad_checkpoint(&state, model, state.size, 1);
    memcpy(state.bytes, original, state.size);
    put_float(&state, 1, 0, 1.0f);
    expect_bad_checkpoint(&state, model, state.size, 1);
    memcpy(state.bytes, original, state.size);
    put_float(&state, 2, 0, 1.0f);
    expect_bad_checkpoint(&state, model, state.size, 1);
    memcpy(state.bytes, original, state.size);
    put_float(&state, 3, 0, -1.0f);
    expect_bad_checkpoint(&state, model, state.size, 1);
    memcpy(state.bytes, original, state.size);
    write_fixture(&state, state.size, 0);
    {
        FILE *file = fopen(fixture_path, "ab");
        cmlp_model *result = model;
        CHECK(file != NULL); CHECK(fputc(0, file) == 0); CHECK(fclose(file) == 0);
        CHECK(cmlp_load(fixture_path, &result) == CMLP_INVALID_FORMAT && result == model);
        CHECK(remove(fixture_path) == 0);
    }
    OK(cmlp_forward(model, input, 1, NULL));
    OK(cmlp_backward(model, gradient, 1)); OK(cmlp_step(model));
    free(original); free(state.bytes); cmlp_destroy(model);
}

static void test_dqn_td_update(void)
{
    const int widths[] = {2, 8, 2};
    const float states[] = {0.3f, -0.2f, -0.5f, 0.6f, 0.8f, 0.1f};
    const float next_states[] = {0.2f, -0.1f, -0.3f, 0.4f, 0.7f, 0.2f};
    const float rewards[] = {0.8f, -0.4f, 0.5f};
    const size_t actions[] = {0, 1, 0};
    const int terminal[] = {0, 1, 0};
    cmlp_config config = cmlp_default_config();
    cmlp_model *online = NULL, *target = NULL;
    float next_q[6], q[6], td_target[3], derivative[6];
    double initial_loss = 0.0, final_loss = 0.0;
    size_t i, iteration;
    config.layers = widths; config.layer_count = 3; config.seed = 99;
    config.hidden_activation = CMLP_RELU; config.output_activation = CMLP_LINEAR;
    config.optimizer = CMLP_ADAM; config.learning_rate = DQN_LEARNING_RATE;
    config.gradient_clip = DQN_GRADIENT_CLIP;
    OK(cmlp_create(&config, &online)); OK(cmlp_create(&config, &target));
    OK(cmlp_copy_from(target, online));
    OK(cmlp_forward(target, next_states, 3, next_q));
    for (i = 0; i < 3; ++i)
        td_target[i] = rewards[i] + (terminal[i] ? 0.0f : 0.95f * fmaxf(next_q[i * 2], next_q[i * 2 + 1]));
    near_value(td_target[1], rewards[1], 0.0);
    for (iteration = 0; iteration < 120; ++iteration) {
        double loss = 0.0;
        OK(cmlp_forward(online, states, 3, q));
        memset(derivative, 0, sizeof(derivative));
        for (i = 0; i < 3; ++i) {
            const float difference = q[2 * i + actions[i]] - td_target[i];
            derivative[2 * i + actions[i]] = difference;
            loss += 0.5 * difference * difference / 3.0;
        }
        if (iteration == 0) initial_loss = loss;
        OK(cmlp_backward(online, derivative, 3));
        OK(cmlp_step(online));
    }
    OK(cmlp_forward(online, states, 3, q));
    for (i = 0; i < 3; ++i) {
        const double difference = (double)q[2 * i + actions[i]] - td_target[i];
        final_loss += 0.5 * difference * difference / 3.0;
    }
    CHECK(final_loss < initial_loss * 0.5);
    OK(cmlp_forward(target, next_states, 3, q));
    for (i = 0; i < 6; ++i) near_value(q[i], next_q[i], 0.0);
    OK(cmlp_copy_from(target, online));
    compare_parameters(target, online, 0.0);
    CHECK(cmlp_step(target) == CMLP_INVALID_STATE);
    cmlp_destroy(online); cmlp_destroy(target);
}

int main(void)
{
    test_initialization_and_rng();
    test_activation_gradients();
    test_cross_entropy();
    test_batching_and_retention();
    test_optimizers();
    test_invalid_inputs();
    test_numeric_failures();
    test_gradient_range_bounds();
    test_persistence_and_copy();
    test_checkpoint_corruption();
    test_dqn_td_update();
    printf("cmlp-tests: %lu checks passed (activations, CE, batching, SGD/Adam, clipping, DQN, checkpoints)\n", checks);
    return EXIT_SUCCESS;
}
