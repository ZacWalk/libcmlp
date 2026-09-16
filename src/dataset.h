#ifndef NN_DATASET_H
#define NN_DATASET_H
#include "cmlp.h"
#include "common.h"

/* Initialize to {0}; reset/load replace owned storage only on success. */
typedef struct dataset {
    int classes;
    int dimensions;
    size_t sample_count;
    size_t capacity;
    xfloat *X;
    xfloat *Y;
} dataset;

cmlp_status dataset_reset(dataset *data, int classes, int dimensions, size_t capacity);
cmlp_status dataset_append(dataset *data, int label, xfloat **features);
cmlp_status dataset_discard_last(dataset *data);
void dataset_free(dataset *data);
const xfloat *dataset_sample_x(const dataset *data, size_t index);
const xfloat *dataset_sample_y(const dataset *data, size_t index);
#endif
