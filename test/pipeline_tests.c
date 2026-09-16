#include "pipeline.h"
#include "trainer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #expr); return 1; \
} } while (0)

static int write_text(const char *path, const char *text)
{
    FILE *stream = fopen(path, "wb");
    int written, closed;
    if (!stream) return 0;
    written = fputs(text, stream) >= 0;
    closed = fclose(stream) == 0;
    return written && closed;
}

static int storage(void)
{
    dataset data = {0};
    xfloat *row;
    CHECK(dataset_reset(&data, 3, 2, 1) == CMLP_OK);
    CHECK(dataset_append(&data, 2, &row) == CMLP_OK);
    row[0] = 0.25f; row[1] = 0.75f;
    CHECK(dataset_sample_y(&data, 0)[2] == 1.0f);
    CHECK(dataset_sample_y(&data, 0)[0] == 0.0f);
    CHECK(dataset_append(&data, 0, &row) == CMLP_OK);
    CHECK(data.sample_count == 2 && data.capacity >= 2);
    CHECK(dataset_sample_x(&data, 0)[1] == 0.75f);
    CHECK(dataset_discard_last(&data) == CMLP_OK && data.sample_count == 1);
    CHECK(dataset_append(&data, 3, &row) == CMLP_INVALID_ARGUMENT);
    CHECK(dataset_reset(&data, 3, 2, SIZE_MAX) == CMLP_OUT_OF_MEMORY);
    CHECK(data.sample_count == 1 && dataset_sample_x(&data, 0)[0] == 0.25f);
    CHECK(dataset_reset(&data, 0, 2, 1) == CMLP_INVALID_ARGUMENT);
    CHECK(dataset_discard_last(&data) == CMLP_OK);
    CHECK(dataset_discard_last(&data) == CMLP_INVALID_STATE);
    dataset_free(&data);
    dataset_free(&data);
    return 0;
}

static int parsing(const char *path)
{
    dataset data = {0};
    CHECK(write_text(path, "label,a,b\r\n0,255,0\r\n9,1,2\r\n0,1\r\n"
        "0,1,2,3\r\n0,1,2,\r\n0,2147483648,0\r\n0,+1,0\r\n"
        "0,1.5,2\r\n0,,2\r\n0,1x,2\r\n\r\n1,0,255"));
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, true) == CMLP_OK);
    CHECK(data.sample_count == 2 && data.dimensions == 2);
    CHECK(data.X[0] == 1.0f && data.X[1] == 0.0f && data.X[3] == 1.0f);
    CHECK(data.Y[0] == 1.0f && data.Y[3] == 1.0f);
    CHECK(pipeline_load_csv(&data, NULL, 2, 255.0f, true) == CMLP_INVALID_ARGUMENT);
    CHECK(pipeline_load_csv(&data, path, 2, NAN, true) == CMLP_INVALID_ARGUMENT);
    CHECK(pipeline_load_csv(&data, path, 2, 0.0f, true) == CMLP_INVALID_ARGUMENT);
    CHECK(write_text(path, "label,a,b\n0,no,1\n"));
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, true) == CMLP_INVALID_FORMAT);
    CHECK(data.sample_count == 2 && data.X[0] == 1.0f);
    CHECK(write_text(path, ""));
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, true) == CMLP_INVALID_FORMAT);
    CHECK(write_text(path, "label,a,b\n"));
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, true) == CMLP_INVALID_FORMAT);
    CHECK(write_text(path, "0,255,0\n1,0,255\n"));
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, false) == CMLP_OK);
    CHECK(data.sample_count == 2);
    {
        FILE *stream = fopen(path, "wb");
        CHECK(stream != NULL);
        CHECK(fputs("0", stream) >= 0);
        for (int i = 0; i < 3000; ++i) CHECK(fputs(",255", stream) >= 0);
        CHECK(fclose(stream) == 0);
    }
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, false) == CMLP_OK);
    CHECK(data.dimensions == 3000 && data.sample_count == 1 && data.X[2999] == 1.0f);
    {
        const char binary[] = "label,a,b\n0,255,0\0,unexpected\n";
        FILE *stream = fopen(path, "wb");
        CHECK(stream != NULL);
        CHECK(fwrite(binary, 1, sizeof(binary) - 1, stream) == sizeof(binary) - 1);
        CHECK(fclose(stream) == 0);
    }
    CHECK(pipeline_load_csv(&data, path, 2, 255.0f, true) == CMLP_INVALID_FORMAT);
    CHECK(data.dimensions == 3000 && data.sample_count == 1);
    dataset_free(&data);
    return 0;
}

static int pairing_and_training(const char *train_path, const char *test_path)
{
    dataset_pipeline pipeline = {0};
    pipeline_config config = {train_path, test_path, 255.0f, true};
    trainer_config training = {2, 2, 0.9f, 12345};
    cmlp_config model_config = cmlp_default_config();
    int layers[] = {2, 3, 2};
    cmlp_model *model = NULL;
    evaluation_metrics metrics;
    CHECK(write_text(train_path, "label,a,b\n0,255,0\n1,0,255\n0,200,0\n"));
    CHECK(write_text(test_path, "label,a,b\n0,255,0\n1,0,255\n"));
    CHECK(pipeline_load(&pipeline, &config, 2) == CMLP_OK);
    CHECK(pipeline.training.sample_count == 3 && pipeline.evaluation.sample_count == 2);
    CHECK(write_text(test_path, "label,a\n0,255\n"));
    CHECK(pipeline_load(&pipeline, &config, 2) == CMLP_INVALID_FORMAT);
    CHECK(pipeline.evaluation.dimensions == 2 && pipeline.evaluation.sample_count == 2);
    model_config.layers = layers;
    model_config.layer_count = 3;
    model_config.seed = 12345;
    CHECK(cmlp_create(&model_config, &model) == CMLP_OK);
    CHECK(trainer_fit(model, &pipeline.training, &training) == CMLP_OK);
    CHECK(fabsf(cmlp_learning_rate(model) - LEARNING_RATE * 0.9f * 0.9f) < 1e-7f);
    CHECK(trainer_evaluate(model, &pipeline.evaluation, &metrics) == CMLP_OK);
    CHECK(metrics.samples == 2 && isfinite(metrics.loss));
    training.epochs = 0;
    CHECK(trainer_fit(model, &pipeline.training, &training) == CMLP_INVALID_ARGUMENT);
    training.epochs = 1; training.batch_size = 0;
    CHECK(trainer_fit(model, &pipeline.training, &training) == CMLP_INVALID_ARGUMENT);
    training.batch_size = 2; training.learning_rate_decay = NAN;
    CHECK(trainer_fit(model, &pipeline.training, &training) == CMLP_INVALID_ARGUMENT);
    pipeline.evaluation.dimensions = 1;
    CHECK(trainer_evaluate(model, &pipeline.evaluation, &metrics) == CMLP_INVALID_ARGUMENT);
    cmlp_destroy(model);
    pipeline_free(&pipeline);
    return 0;
}

static int large_finite_losses(void)
{
    int layers[] = {1, 2};
    const float input[] = {0.0f};
    const float derivative[] = {-2e38f, 2e38f};
    cmlp_config config = cmlp_default_config();
    cmlp_model *model = NULL;
    dataset data = {0};
    evaluation_metrics metrics;
    cmlp_metrics sample;
    xfloat *row;
    trainer_config training = {1, 2, 1.0f, 12345};
    config.layers = layers;
    config.layer_count = 2;
    config.seed = 12345;
    config.learning_rate = 1.0f;
    config.momentum = 0.0f;
    CHECK(cmlp_create(&config, &model) == CMLP_OK);
    CHECK(cmlp_forward(model, input, 1, NULL) == CMLP_OK);
    CHECK(cmlp_backward(model, derivative, 1) == CMLP_OK);
    CHECK(cmlp_step(model) == CMLP_OK);
    CHECK(dataset_reset(&data, 2, 1, 2) == CMLP_OK);
    CHECK(dataset_append(&data, 1, &row) == CMLP_OK);
    row[0] = 0.0f;
    CHECK(dataset_append(&data, 1, &row) == CMLP_OK);
    row[0] = 0.0f;
    CHECK(cmlp_evaluate_sample(model, input, data.Y, &sample) == CMLP_OK);
    CHECK(sample.loss > 1.8e38f && isfinite(sample.loss));
    CHECK(trainer_evaluate(model, &data, &metrics) == CMLP_OK);
    CHECK(isfinite(metrics.loss) && metrics.loss == sample.loss);
    CHECK(trainer_fit(model, &data, &training) == CMLP_OK);
    CHECK(trainer_evaluate(model, &data, &metrics) == CMLP_OK);
    CHECK(isfinite(metrics.loss) && metrics.loss == sample.loss);
    cmlp_destroy(model);
    dataset_free(&data);
    return 0;
}

int main(void)
{
    const char *train_path = "pipeline-test-train.csv";
    const char *test_path = "pipeline-test-evaluation.csv";
    int result = storage();
    if (!result) result = parsing(train_path);
    if (!result) result = pairing_and_training(train_path, test_path);
    if (!result) result = large_finite_losses();
    /* Both names belong exclusively to this test in its build-tree scratch dir. */
    if (remove(train_path) != 0 && result == 0) result = 1;
    if (remove(test_path) != 0 && result == 0) result = 1;
    if (!result) puts("\nPipeline and trainer tests passed");
    return result;
}
