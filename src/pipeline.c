#include "pipeline.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct csv_reader {
    FILE *stream;
    char *buffer;
    size_t position;
    size_t length;
} csv_reader;

static cmlp_status read_line(csv_reader *reader, char **line, size_t *capacity, bool *present)
{
    size_t length = 0;
    *present = false;
    for (;;) {
        const char *begin;
        const char *newline;
        size_t take;
        if (reader->position == reader->length) {
            reader->length = fread(reader->buffer, 1, 1u << 20, reader->stream);
            reader->position = 0;
            if (!reader->length) {
                if (ferror(reader->stream)) return CMLP_IO_ERROR;
                if (!length) return CMLP_OK;
                break;
            }
        }
        begin = reader->buffer + reader->position;
        newline = (const char *)memchr(begin, '\n', reader->length - reader->position);
        take = newline ? (size_t)(newline - begin) + 1 : reader->length - reader->position;
        if (memchr(begin, '\0', take)) return CMLP_INVALID_FORMAT;
        if (length > SIZE_MAX - take - 1) return CMLP_OUT_OF_MEMORY;
        if (length + take + 1 > *capacity) {
            size_t new_capacity = *capacity ? *capacity : 4096;
            char *larger;
            while (new_capacity < length + take + 1) {
                if (new_capacity > SIZE_MAX / 2) return CMLP_OUT_OF_MEMORY;
                new_capacity *= 2;
            }
            larger = (char *)realloc(*line, new_capacity);
            if (!larger) return CMLP_OUT_OF_MEMORY;
            *line = larger;
            *capacity = new_capacity;
        }
        memcpy(*line + length, begin, take);
        length += take;
        reader->position += take;
        if (newline) break;
    }
    while (length && ((*line)[length - 1] == '\n' || (*line)[length - 1] == '\r'))
        --length;
    (*line)[length] = '\0';
    *present = true;
    return CMLP_OK;
}

static bool parse_integer(const char **cursor, int *value, bool *more)
{
    const char *p = *cursor;
    unsigned int magnitude = 0;
    bool negative = *p == '-';
    unsigned int limit = negative ? (unsigned int)INT_MAX + 1u : (unsigned int)INT_MAX;
    if (negative) ++p;
    if (*p < '0' || *p > '9') return false;
    do {
        unsigned int digit = (unsigned int)(*p - '0');
        if (magnitude > (limit - digit) / 10u) return false;
        magnitude = magnitude * 10u + digit;
        ++p;
    } while (*p >= '0' && *p <= '9');
    if (*p != ',' && *p != '\0') return false;
    *value = negative ? (magnitude == (unsigned int)INT_MAX + 1u ? INT_MIN : -(int)magnitude)
                      : (int)magnitude;
    *more = *p == ',';
    *cursor = *more ? p + 1 : p;
    return true;
}

static cmlp_status parse_row(dataset *data, const char *line, xfloat scale,
    const char *filename, size_t row)
{
    const char *cursor = line;
    int label;
    bool more;
    xfloat *features;
    cmlp_status status;
    if (!*line) return CMLP_OK;
    if (!parse_integer(&cursor, &label, &more) || label < 0 || label >= data->classes) {
        fprintf(stderr, "Warning: Skipping malformed label in %s at row %zu\n", filename, row);
        return CMLP_OK;
    }
    status = dataset_append(data, label, &features);
    if (status != CMLP_OK) return status;
    for (int i = 0; i < data->dimensions; ++i) {
        int pixel;
        if (!more || !parse_integer(&cursor, &pixel, &more)) goto malformed;
        features[i] = (xfloat)pixel * scale;
        if (!isfinite(features[i])) goto malformed;
    }
    if (!more) return CMLP_OK;
malformed:
    fprintf(stderr, "Warning: Skipping malformed row in %s at row %zu\n", filename, row);
    return dataset_discard_last(data);
}

cmlp_status pipeline_load_csv(dataset *target, const char *filename, int classes,
    xfloat x_max, bool has_header)
{
    FILE *stream = NULL;
    char *stream_buffer = NULL;
    char *line = NULL;
    size_t line_capacity = 0, sample_capacity = 0, row = 1;
    int dimensions = 0;
    bool present;
    dataset next = {0};
    cmlp_status status = CMLP_OK;
    csv_reader reader = {0};
    long file_size;
    if (!target || !filename || !*filename || classes <= 0 ||
        !isfinite(x_max) || x_max <= 0.0f || !isfinite(1.0f / x_max)) {
        fprintf(stderr, "Error: Invalid CSV configuration\n");
        return CMLP_INVALID_ARGUMENT;
    }
    stream = fopen(filename, "rb");
    if (!stream) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return CMLP_IO_ERROR;
    }
    stream_buffer = (char *)malloc(1u << 20);
    if (!stream_buffer) { status = CMLP_OUT_OF_MEMORY; goto done; }
    if (setvbuf(stream, NULL, _IONBF, 0) != 0) {
        status = CMLP_IO_ERROR;
        goto done;
    }
    reader.stream = stream;
    reader.buffer = stream_buffer;
    /* Regular CSV files permit an up-front reserve without per-row reallocations. */
    if (fseek(stream, 0, SEEK_END) != 0 || (file_size = ftell(stream)) < 0 ||
        fseek(stream, 0, SEEK_SET) != 0) {
        status = CMLP_IO_ERROR;
        goto done;
    }
    printf("Loading %s\n", filename);
    status = read_line(&reader, &line, &line_capacity, &present);
    if (status != CMLP_OK) goto done;
    if (!present) { status = CMLP_INVALID_FORMAT; goto done; }
    for (const char *p = line; *p; ++p) {
        if (*p == ',') {
            if (dimensions == INT_MAX) { status = CMLP_INVALID_FORMAT; goto done; }
            ++dimensions;
        }
    }
    if (dimensions <= 0) { status = CMLP_INVALID_FORMAT; goto done; }
    if (has_header) {
        ++row;
        status = read_line(&reader, &line, &line_capacity, &present);
        if (status != CMLP_OK) goto done;
        if (!present) { status = CMLP_INVALID_FORMAT; goto done; }
    }
    if (*line) sample_capacity = (size_t)file_size / (strlen(line) + 1);
    status = dataset_reset(&next, classes, dimensions, sample_capacity);
    if (status != CMLP_OK) goto done;
    do {
        status = parse_row(&next, line, 1.0f / x_max, filename, row);
        if (status != CMLP_OK) goto done;
        ++row;
        status = read_line(&reader, &line, &line_capacity, &present);
        if (status != CMLP_OK) goto done;
    } while (present);
    if (!next.sample_count) status = CMLP_INVALID_FORMAT;
done:
    if (fclose(stream) != 0 && status == CMLP_OK) status = CMLP_IO_ERROR;
    free(stream_buffer);
    free(line);
    if (status == CMLP_OK) {
        dataset_free(target);
        *target = next;
    } else {
        fprintf(stderr, "Error: Loading %s: %s\n", filename, cmlp_status_string(status));
        dataset_free(&next);
    }
    return status;
}

cmlp_status pipeline_load(dataset_pipeline *pipeline, const pipeline_config *config, int classes)
{
    dataset_pipeline next = {0};
    cmlp_status status;
    if (!pipeline || !config) {
        fprintf(stderr, "Error: Missing dataset pipeline configuration\n");
        return CMLP_INVALID_ARGUMENT;
    }
    status = pipeline_load_csv(&next.training, config->training_file, classes,
        config->x_max, config->has_header);
    if (status == CMLP_OK)
        status = pipeline_load_csv(&next.evaluation, config->evaluation_file, classes,
            config->x_max, config->has_header);
    if (status == CMLP_OK && next.training.dimensions != next.evaluation.dimensions) {
        fprintf(stderr, "Error: Training and test datasets have inconsistent dimensions.\n");
        status = CMLP_INVALID_FORMAT;
    }
    if (status != CMLP_OK) {
        pipeline_free(&next);
        return status;
    }
    pipeline_free(pipeline);
    *pipeline = next;
    return CMLP_OK;
}

void pipeline_free(dataset_pipeline *pipeline)
{
    if (!pipeline) return;
    dataset_free(&pipeline->training);
    dataset_free(&pipeline->evaluation);
}
