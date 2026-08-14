// Training and evaluation orchestration types for epoch-level batching and aggregate metrics.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.h"

class dataset;
class nn;

struct training_epoch_metrics
{
    xfloat loss = 0.0f;
    std::size_t correct = 0;
};

struct evaluation_metrics
{
    xfloat loss = 0.0f;
    std::size_t correct = 0;
    std::size_t samples = 0;
};

struct trainer_config
{
    int epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    xfloat learning_rate_decay = LR_DECAY; // multiplied into the learning rate after each epoch
    std::uint32_t seed = 0; // 0 draws a nondeterministic shuffle order
};

class trainer
{
public:
    explicit trainer(trainer_config config = {}) : config(config)
    {
    }

    std::vector<training_epoch_metrics> fit(nn& model, const dataset& data) const;
    evaluation_metrics evaluate(nn& model, const dataset& data) const;

private:
    trainer_config config;
};