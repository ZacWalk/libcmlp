
// Dataset storage helpers for reserving, appending, and discarding parsed samples.

#include "dataset.h"

#include <stdexcept>

void dataset::reset(const int input_dimensions, const std::size_t sample_capacity)
{
	if (input_dimensions <= 0)
	{
		throw std::invalid_argument("Dataset dimensions must be positive");
	}

	dimensions = input_dimensions;
	sample_count = 0;
	X.clear();
	Y.clear();
	if (sample_capacity > 0)
	{
		X.reserve(sample_capacity * static_cast<std::size_t>(dimensions));
		Y.reserve(sample_capacity * static_cast<std::size_t>(classes));
	}
}

xfloat* dataset::append_sample(const int label)
{
	if (dimensions <= 0)
	{
		throw std::logic_error("Dataset must be reset before appending samples");
	}

	if (label < 0 || label >= classes)
	{
		throw std::out_of_range("Sample label is out of range");
	}

	const std::size_t x_base = X.size();
	const std::size_t y_base = Y.size();
	X.resize(x_base + static_cast<std::size_t>(dimensions));
	Y.resize(y_base + static_cast<std::size_t>(classes), 0.0f);
	Y[y_base + static_cast<std::size_t>(label)] = 1.0f;
	sample_count += 1;
	return X.data() + x_base;
}

void dataset::discard_last_sample(void)
{
	if (sample_count == 0)
	{
		throw std::logic_error("No samples are available to discard");
	}

	X.resize(X.size() - static_cast<std::size_t>(dimensions));
	Y.resize(Y.size() - static_cast<std::size_t>(classes));
	sample_count -= 1;
}