#include "dataset.h"

#include <stdlib.h>
#include <string.h>

static cmlp_status reserve(dataset *data, size_t capacity)
{
    xfloat *x;
    xfloat *y;
    if (capacity <= data->capacity) return CMLP_OK;
    if (capacity > SIZE_MAX / sizeof(xfloat) / (size_t)data->dimensions ||
        capacity > SIZE_MAX / sizeof(xfloat) / (size_t)data->classes)
        return CMLP_OUT_OF_MEMORY;
    x = (xfloat *)realloc(data->X, capacity * (size_t)data->dimensions * sizeof(xfloat));
    if (!x) return CMLP_OUT_OF_MEMORY;
    data->X = x;
    y = (xfloat *)realloc(data->Y, capacity * (size_t)data->classes * sizeof(xfloat));
    if (!y) return CMLP_OUT_OF_MEMORY;
    data->Y = y;
    data->capacity = capacity;
    return CMLP_OK;
}

cmlp_status dataset_reset(dataset *data, int classes, int dimensions, size_t capacity)
{
    dataset next = {0};
    cmlp_status status;
    if (!data || classes <= 0 || dimensions <= 0) return CMLP_INVALID_ARGUMENT;
    next.classes = classes;
    next.dimensions = dimensions;
    status = reserve(&next, capacity);
    if (status != CMLP_OK) {
        dataset_free(&next);
        return status;
    }
    dataset_free(data);
    *data = next;
    return CMLP_OK;
}

cmlp_status dataset_append(dataset *data, int label, xfloat **features)
{
    cmlp_status status;
    xfloat *target;
    size_t capacity;
    if (!data || !features || data->dimensions <= 0 || data->classes <= 0 ||
        label < 0 || label >= data->classes)
        return CMLP_INVALID_ARGUMENT;
    if (data->sample_count == data->capacity) {
        if (data->capacity > SIZE_MAX / 2) return CMLP_OUT_OF_MEMORY;
        capacity = data->capacity ? data->capacity * 2 : 256;
        status = reserve(data, capacity);
        if (status != CMLP_OK) return status;
    }
    *features = data->X + data->sample_count * (size_t)data->dimensions;
    target = data->Y + data->sample_count * (size_t)data->classes;
    memset(target, 0, (size_t)data->classes * sizeof(*target));
    target[label] = 1.0f;
    ++data->sample_count;
    return CMLP_OK;
}

cmlp_status dataset_discard_last(dataset *data)
{
    if (!data || !data->sample_count) return CMLP_INVALID_STATE;
    --data->sample_count;
    return CMLP_OK;
}

void dataset_free(dataset *data)
{
    if (!data) return;
    free(data->X);
    free(data->Y);
    memset(data, 0, sizeof(*data));
}

const xfloat *dataset_sample_x(const dataset *data, size_t index)
{
    return data->X + index * (size_t)data->dimensions;
}

const xfloat *dataset_sample_y(const dataset *data, size_t index)
{
    return data->Y + index * (size_t)data->classes;
}
