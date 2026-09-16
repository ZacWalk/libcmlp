#include "trainer.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "random.h"

static int compatible(const cmlp_model *model, const dataset *data)
{
    return model && data && data->sample_count && data->X && data->Y &&
        data->dimensions > 0 && data->classes > 0 &&
        cmlp_input_size(model) == (size_t)data->dimensions &&
        cmlp_output_size(model) == (size_t)data->classes;
}

cmlp_status trainer_fit(cmlp_model *model, const dataset *data, const trainer_config *config)
{
    nn_rng rng;
    size_t *indices;
    cmlp_status status = CMLP_OK;
    if (!compatible(model, data) || !config || config->epochs <= 0 ||
        config->batch_size <= 0 || !isfinite(config->learning_rate_decay) ||
        config->learning_rate_decay <= 0.0f || data->sample_count > UINT32_MAX ||
        data->sample_count > SIZE_MAX / sizeof(*indices))
        return CMLP_INVALID_ARGUMENT;
    indices = (size_t *)malloc(data->sample_count * sizeof(*indices));
    if (!indices) return CMLP_OUT_OF_MEMORY;
    for (size_t i = 0; i < data->sample_count; ++i) indices[i] = i;
    nn_rng_init(&rng, config->seed ? config->seed : nn_random_seed());
    status = cmlp_zero_grad(model);
    if (status != CMLP_OK) goto done;
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        double loss = 0.0;
        size_t correct = 0, batch_count = 0;
        for (size_t i = data->sample_count; i > 1; --i) {
            size_t j = nn_rng_bounded(&rng, (uint32_t)i);
            size_t temp = indices[i - 1];
            indices[i - 1] = indices[j];
            indices[j] = temp;
        }
        for (size_t i = 0; i < data->sample_count; ++i) {
            size_t sample = indices[i];
            cmlp_metrics metrics;
            status = cmlp_train_sample(model, dataset_sample_x(data, sample),
                dataset_sample_y(data, sample), &metrics);
            if (status != CMLP_OK) goto done;
            loss += metrics.loss;
            correct += metrics.correct;
            if (++batch_count == (size_t)config->batch_size) {
                status = cmlp_step(model);
                if (status != CMLP_OK) goto done;
                batch_count = 0;
            }
        }
        if (batch_count) {
            status = cmlp_step(model);
            if (status != CMLP_OK) goto done;
        }
        if (config->learning_rate_decay != 1.0f) {
            status = cmlp_scale_learning_rate(model, config->learning_rate_decay);
            if (status != CMLP_OK) goto done;
        }
        loss /= (double)data->sample_count;
        if (!isfinite(loss) || loss < 0.0 || loss > FLT_MAX) {
            status = CMLP_NUMERIC_ERROR;
            goto done;
        }
        printf("\n[EPOCH %4d] [LOSS %.5f] [ACCURACY %6zu out of %zu]",
            epoch + 1, loss, correct, data->sample_count);
    }
done:
    free(indices);
    return status;
}

cmlp_status trainer_evaluate(cmlp_model *model, const dataset *data, evaluation_metrics *metrics)
{
    evaluation_metrics result = {0};
    double loss = 0.0;
    if (!compatible(model, data) || !metrics) return CMLP_INVALID_ARGUMENT;
    result.samples = data->sample_count;
    for (size_t i = 0; i < data->sample_count; ++i) {
        cmlp_metrics sample;
        cmlp_status status = cmlp_evaluate_sample(model, dataset_sample_x(data, i),
            dataset_sample_y(data, i), &sample);
        if (status != CMLP_OK) return status;
        loss += sample.loss;
        result.correct += sample.correct;
    }
    loss /= (double)result.samples;
    if (!isfinite(loss) || loss < 0.0 || loss > FLT_MAX) return CMLP_NUMERIC_ERROR;
    result.loss = (xfloat)loss;
    *metrics = result;
    printf("\n\n[EVALUATION] [LOSS %.5f] [ACCURACY %6zu out of %zu]",
        result.loss, result.correct, result.samples);
    return CMLP_OK;
}
