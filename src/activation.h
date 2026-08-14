// Activation functions and derivatives used by the multilayer perceptron.

#pragma once

#include <cmath>

#include "common.h"

__forceinline xfloat sigmoid(const xfloat x)
{
    if (x >= 0.0f)
    {
        const xfloat exponent = std::exp(-x);
        return 1.0f / (1.0f + exponent);
    }

    const xfloat exponent = std::exp(x);
    return exponent / (1.0f + exponent);
}

// Takes the already-activated value, so the derivative is a single multiply.
__forceinline constexpr xfloat sig_derivative(const xfloat x)
{
    return x * (1.0f - x);
}
