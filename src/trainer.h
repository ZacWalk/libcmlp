#ifndef NN_TRAINER_H
#define NN_TRAINER_H
#include "cmlp.h"
#include "common.h"
#include "dataset.h"

typedef struct evaluation_metrics {
    xfloat loss;
    size_t correct;
    size_t samples;
} evaluation_metrics;

typedef struct trainer_config {
    int epochs;
    int batch_size;
    xfloat learning_rate_decay;
    uint32_t seed;
} trainer_config;

cmlp_status trainer_fit(cmlp_model *model, const dataset *data, const trainer_config *config);
cmlp_status trainer_evaluate(cmlp_model *model, const dataset *data, evaluation_metrics *metrics);
#endif