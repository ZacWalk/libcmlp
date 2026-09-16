#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <math.h>
#include "common.h"
#include "cmlp.h"

static NN_FORCEINLINE float nn_sigmoid(float x)
{
    const float e = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

static NN_FORCEINLINE float nn_activate(cmlp_activation kind, float x)
{
    switch (kind) {
    case CMLP_SIGMOID: return nn_sigmoid(x);
    case CMLP_RELU: return x > 0.0f ? x : 0.0f;
    case CMLP_TANH: return tanhf(x);
    default: return x;
    }
}

/* Derivatives use the retained activation; ReLU uses derivative zero at zero. */
static NN_FORCEINLINE float nn_activation_derivative(cmlp_activation kind, float a)
{
    switch (kind) {
    case CMLP_SIGMOID: return a * (1.0f - a);
    case CMLP_RELU: return a > 0.0f ? 1.0f : 0.0f;
    case CMLP_TANH: return (1.0f - a) * (1.0f + a);
    default: return 1.0f;
    }
}
#endif
