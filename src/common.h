#ifndef NN_COMMON_H
#define NN_COMMON_H
#include <stddef.h>
#include <stdint.h>

typedef float xfloat;

#if defined(_MSC_VER)
#define NN_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define NN_FORCEINLINE inline __attribute__((always_inline))
#else
#define NN_FORCEINLINE inline
#endif

#define EPOCHS 30
#define BATCH_SIZE 16
#define HIDDEN_1 100
#define HIDDEN_2 50
#define MNIST_CLASSES 10
#define LEARNING_RATE 0.08f
#define MOMENTUM 0.9f
#define LR_DECAY 0.90f
#define MNIST_MAX_VAL 255.0f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPSILON 1e-8f
#define DQN_LEARNING_RATE 0.001f
#define DQN_GRADIENT_CLIP 10.0f
#define DEFAULT_TRAIN_CSV "data/fashion-mnist_train.csv"
#define DEFAULT_TEST_CSV "data/fashion-mnist_test.csv"
#endif
