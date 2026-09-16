#ifndef NN_PIPELINE_H
#define NN_PIPELINE_H
#include <stdbool.h>
#include "dataset.h"

typedef struct pipeline_config {
    const char *training_file;
    const char *evaluation_file;
    xfloat x_max;
    bool has_header;
} pipeline_config;

typedef struct dataset_pipeline {
    dataset training;
    dataset evaluation;
} dataset_pipeline;

cmlp_status pipeline_load(dataset_pipeline *pipeline, const pipeline_config *config, int classes);
cmlp_status pipeline_load_csv(dataset *target, const char *filename, int classes,
    xfloat x_max, bool has_header);
void pipeline_free(dataset_pipeline *pipeline);
#endif