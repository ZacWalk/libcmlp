#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "cmlp.h"
#include "pipeline.h"
#include "trainer.h"

static bool read_env_int(const char *name, int fallback, int *out)
{
    const char *value = getenv(name);
    char *end;
    long parsed;
    if (!value) { *out = fallback; return true; }
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (end == value || *end || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) {
        fprintf(stderr, "Error: %s must be an integer in [%d, %d]\n", name, INT_MIN, INT_MAX);
        return false;
    }
    *out = (int)parsed;
    return true;
}

static bool read_env_float(const char *name, xfloat fallback, xfloat *out)
{
    const char *value = getenv(name);
    char *end;
    xfloat parsed;
    if (!value) { *out = fallback; return true; }
    errno = 0;
    parsed = strtof(value, &end);
    if (end == value || *end || errno == ERANGE || !isfinite(parsed)) {
        fprintf(stderr, "Error: %s must be a finite number\n", name);
        return false;
    }
    *out = parsed;
    return true;
}

static bool read_seed(uint32_t *seed)
{
    const char *value = getenv("NN_SEED");
    uint32_t parsed = 0;
    if (!value) { *seed = 0; return true; }
    if (!*value) goto invalid;
    for (const char *p = value; *p; ++p) {
        uint32_t digit;
        if (*p < '0' || *p > '9') goto invalid;
        digit = (uint32_t)(*p - '0');
        if (parsed > (UINT32_MAX - digit) / 10u) goto invalid;
        parsed = parsed * 10u + digit;
    }
    *seed = parsed;
    return true;
invalid:
    fprintf(stderr, "Error: NN_SEED must be an integer in [0, 4294967295]\n");
    return false;
}

static const char *read_path(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return value && *value ? value : fallback;
}

static bool monotonic_seconds(double *seconds)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter)) return false;
    *seconds = (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return false;
    *seconds = (double)value.tv_sec + (double)value.tv_nsec / 1e9;
#endif
    return true;
}

int main(int argc, char **argv)
{
    double start, end;
    int layers[4] = {0, HIDDEN_1, HIDDEN_2, MNIST_CLASSES};
    cmlp_config model_config = cmlp_default_config();
    trainer_config training_config = {EPOCHS, BATCH_SIZE, LR_DECAY, 0};
    pipeline_config data_config = {
        read_path("NN_TRAIN_CSV", DEFAULT_TRAIN_CSV),
        read_path("NN_TEST_CSV", DEFAULT_TEST_CSV),
        MNIST_MAX_VAL, true
    };
    dataset_pipeline pipeline = {0};
    cmlp_model *model = NULL;
    evaluation_metrics metrics;
    cmlp_status status;
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        puts("mlp-cli: train and evaluate the Fashion-MNIST classifier\n"
             "Usage: mlp-cli [--help]\n"
             "Configuration: NN_TRAIN_CSV, NN_TEST_CSV, NN_HIDDEN1, NN_HIDDEN2,\n"
             "NN_LR, NN_MOMENTUM, NN_LR_DECAY, NN_BATCH_SIZE, NN_EPOCHS, NN_SEED.");
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "Error: Unknown arguments; use mlp-cli --help\n");
        return 1;
    }
    if (!monotonic_seconds(&start)) {
        fprintf(stderr, "Error: Unable to read monotonic clock\n");
        return 1;
    }
    if (!read_env_int("NN_HIDDEN1", HIDDEN_1, &layers[1]) ||
        !read_env_int("NN_HIDDEN2", HIDDEN_2, &layers[2]) ||
        !read_env_int("NN_EPOCHS", EPOCHS, &training_config.epochs) ||
        !read_env_int("NN_BATCH_SIZE", BATCH_SIZE, &training_config.batch_size) ||
        !read_env_float("NN_LR", LEARNING_RATE, &model_config.learning_rate) ||
        !read_env_float("NN_MOMENTUM", MOMENTUM, &model_config.momentum) ||
        !read_env_float("NN_LR_DECAY", LR_DECAY, &training_config.learning_rate_decay) ||
        !read_seed(&model_config.seed))
        return 1;
    if (layers[1] <= 0 || layers[2] <= 0 || training_config.epochs <= 0 ||
        training_config.batch_size <= 0 || training_config.learning_rate_decay <= 0.0f ||
        model_config.learning_rate <= 0.0f ||
        model_config.momentum < 0.0f || model_config.momentum >= 1.0f) {
        fprintf(stderr, "Error: Layer widths, epochs, batch size, learning rate and decay must "
            "be positive; momentum must be in [0, 1)\n");
        return 1;
    }
    training_config.seed = model_config.seed;
    status = pipeline_load(&pipeline, &data_config, MNIST_CLASSES);
    if (status != CMLP_OK) return 1;
    layers[0] = pipeline.training.dimensions;
    model_config.layers = layers;
    model_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    status = cmlp_create(&model_config, &model);
    if (status != CMLP_OK) goto done;
    printf("\n\nNeural Network Summary:\t\t[hidden := Sigmoid, output := Softmax]\n\n");
    for (size_t i = 0; i < model_config.layer_count; ++i)
        printf("Layer %zu\t%4d neurons%s\n", i + 1, layers[i], i ? " + bias" : "");
    printf("Parameters\t%zu trainable weights and biases\nLearning Rate\t%g\nMomentum\t%g\n",
        cmlp_parameter_count(model), (double)model_config.learning_rate, (double)model_config.momentum);
    status = trainer_fit(model, &pipeline.training, &training_config);
    if (status == CMLP_OK) status = trainer_evaluate(model, &pipeline.evaluation, &metrics);
done:
    cmlp_destroy(model);
    pipeline_free(&pipeline);
    if (status != CMLP_OK) {
        fprintf(stderr, "\nTraining failed: %s\n", cmlp_status_string(status));
        return 1;
    }
    if (!monotonic_seconds(&end)) {
        fprintf(stderr, "Error: Unable to read monotonic clock\n");
        return 1;
    }
    printf("\n\nTime taken: %.5f seconds\n", end - start);
    return 0;
}
