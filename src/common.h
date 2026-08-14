
// Shared scalar aliases and project-wide constants for the Fashion-MNIST training pipeline.

#pragma once
#include <cstddef>
#include <cstdint>

using xfloat = float;

// Single source of truth for the tuned defaults; every value is overridable at runtime (see main.cpp).
constexpr int EPOCHS = 30;
constexpr int BATCH_SIZE = 16;
constexpr int HIDDEN_1 = 100;
constexpr int HIDDEN_2 = 50;
constexpr int MNIST_CLASSES = 10;
constexpr xfloat LEARNING_RATE = 0.08f;
constexpr xfloat MOMENTUM = 0.9f;
constexpr xfloat LR_DECAY = 0.90f;
constexpr xfloat MNIST_MAX_VAL = 255.0f;

