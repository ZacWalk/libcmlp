// CSV parsing and dataset loading logic for normalized Fashion-MNIST training and evaluation data.

#include "pipeline.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
int count_columns(const std::string_view line)
{
	if (line.empty())
	{
		return 0;
	}

	return static_cast<int>(std::count(line.begin(), line.end(), ',')) + 1;
}

bool next_token(const std::string_view line, std::size_t& start, std::string_view& token)
{
	if (start == std::string_view::npos)
	{
		return false;
	}

	const std::size_t end = line.find(',', start);
	if (end == std::string_view::npos)
	{
		token = line.substr(start);
		start = std::string_view::npos;
		return true;
	}

	token = line.substr(start, end - start);
	start = end + 1;
	return true;
}

bool parse_int(const std::string_view token, int& value)
{
	const char* const begin = token.data();
	const char* const end = begin + token.size();
	const auto result = std::from_chars(begin, end, value);
	return result.ec == std::errc{} && result.ptr == end;
}

// Reserve capacity up front from the file size and a representative data row.
std::size_t estimate_samples(const char* filename, const std::size_t row_length)
{
	if (row_length == 0)
	{
		return 0;
	}

	std::error_code error;
	const auto file_size = std::filesystem::file_size(filename, error);
	return error ? 0 : file_size / (row_length + 1);
}
}

dataset_pipeline::dataset_pipeline(const int classes) : training(classes), evaluation(classes)
{
}

bool dataset_pipeline::load(const pipeline_config& config)
{
	if (!load_csv(training, config.training_file, config.x_max, config.has_header))
	{
		return false;
	}

	if (!load_csv(evaluation, config.evaluation_file, config.x_max, config.has_header))
	{
		return false;
	}

	return validate_pair();
}

int dataset_pipeline::input_dimensions(void) const
{
	return training.dimensions;
}

const dataset& dataset_pipeline::training_data(void) const
{
	return training;
}

const dataset& dataset_pipeline::evaluation_data(void) const
{
	return evaluation;
}

bool dataset_pipeline::load_csv(dataset& target, const char* filename, const xfloat x_max, const bool has_header) const
{
	if (filename == nullptr)
	{
		std::cerr << "Error: Missing dataset filename" << std::endl;
		return false;
	}

	if (x_max <= 0.0f)
	{
		std::cerr << "Error: x_max must be positive when loading " << filename << std::endl;
		return false;
	}

	// The buffer must outlive the stream and be installed before the file is opened.
	std::vector<char> stream_buffer(1 << 20);
	std::ifstream stream;
	stream.rdbuf()->pubsetbuf(stream_buffer.data(), static_cast<std::streamsize>(stream_buffer.size()));
	stream.open(filename);
	if (!stream.is_open())
	{
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return false;
	}

	std::cout << "Loading " << filename << std::endl;

	std::string line;
	if (!std::getline(stream, line))
	{
		std::cerr << "Error: File is empty " << filename << std::endl;
		return false;
	}

	const int columns = count_columns(line);
	const int input_dimensions = columns - 1;
	if (input_dimensions <= 0)
	{
		std::cerr << "Error: Invalid CSV format in " << filename << std::endl;
		return false;
	}

	if (has_header && !std::getline(stream, line))
	{
		std::cerr << "Error: File contains no data rows " << filename << std::endl;
		return false;
	}

	target.reset(input_dimensions, estimate_samples(filename, line.size()));
	const xfloat scale = 1.0f / x_max;

	const auto parse_row = [&](const std::string_view row) {
		if (row.empty())
		{
			return;
		}

		std::size_t start = 0;
		std::string_view token;
		next_token(row, start, token);

		int label = 0;
		if (!parse_int(token, label) || label < 0 || label >= target.classes)
		{
			std::cerr << "Warning: Skipping malformed label in " << filename << std::endl;
			return;
		}

		auto* features = target.append_sample(label);
		int feature_index = 0;
		while (next_token(row, start, token))
		{
			int pixel = 0;
			if (feature_index >= input_dimensions || !parse_int(token, pixel))
			{
				std::cerr << "Warning: Skipping malformed row in " << filename << std::endl;
				target.discard_last_sample();
				return;
			}

			features[feature_index] = static_cast<xfloat>(pixel) * scale;
			feature_index += 1;
		}

		if (feature_index != input_dimensions)
		{
			std::cerr << "Warning: Skipping row with missing columns in " << filename << std::endl;
			target.discard_last_sample();
		}
	};

	parse_row(line);
	while (std::getline(stream, line))
	{
		parse_row(line);
	}

	return target.samples() > 0;
}

bool dataset_pipeline::validate_pair(void) const
{
	if (training.samples() == 0 || evaluation.samples() == 0)
	{
		std::cerr << "Dataset loading failed." << std::endl;
		return false;
	}

	if (training.dimensions != evaluation.dimensions)
	{
		std::cerr << "Training and test datasets have inconsistent dimensions." << std::endl;
		return false;
	}

	return true;
}