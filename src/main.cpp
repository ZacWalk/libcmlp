
// Entry point that loads Fashion-MNIST CSV data, configures the MLP, and runs training plus evaluation.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

#include "nn.h"
#include "pipeline.h"
#include "trainer.h"

constexpr char training_data_file[] = ".\\data\\fashion-mnist_train.csv";
constexpr char evaluation_data_file[] = ".\\data\\fashion-mnist_test.csv";

namespace
{
int read_env_int(const char* name, const int fallback)
{
    if (const char* value = std::getenv(name))
    {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0')
        {
            return static_cast<int>(parsed);
        }
    }

    return fallback;
}

xfloat read_env_float(const char* name, const xfloat fallback)
{
    if (const char* value = std::getenv(name))
    {
        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end != value && *end == '\0')
        {
            return parsed;
        }
    }

    return fallback;
}
}

int main()
{
    const auto start = std::chrono::steady_clock::now();
    dataset_pipeline pipeline(MNIST_CLASSES);

    if (!pipeline.load({ training_data_file, evaluation_data_file, MNIST_MAX_VAL, true }))
    {
        return 1;
    }

    const auto seed = static_cast<std::uint32_t>(read_env_int("NN_SEED", 0));
    const nn_config model_config{
        { pipeline.input_dimensions(), read_env_int("NN_HIDDEN1", HIDDEN_1), read_env_int("NN_HIDDEN2", HIDDEN_2), MNIST_CLASSES },
        read_env_float("NN_LR", LEARNING_RATE),
        read_env_float("NN_MOMENTUM", MOMENTUM),
        seed,
    };
    const trainer_config training_config{
        read_env_int("NN_EPOCHS", EPOCHS),
        read_env_int("NN_BATCH_SIZE", BATCH_SIZE),
        read_env_float("NN_LR_DECAY", LR_DECAY),
        seed,
    };

    try
    {
        nn fcn;
        fcn.compile(model_config);
        fcn.summary();

        const trainer model_trainer(training_config);
        model_trainer.fit(fcn, pipeline.training_data());
        model_trainer.evaluate(fcn, pipeline.evaluation_data());
    }
    catch (const std::exception& ex)
    {
        std::cerr << "\nTraining failed: " << ex.what() << std::endl;
        return 1;
    }

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "\n\nTime taken: " << elapsed.count() << " seconds\n";

    return 0;
}
