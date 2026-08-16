# Drives the real nn binary and asserts on its console output. There is no test framework
# here by design; ctest invokes this once per case via `cmake -P`.
#
# Required: NN_EXE WORK_DIR CASE
# Optional: TRAIN_CSV TEST_CSV EPOCHS SEED EXPECT_TRAIN EXPECT_TOTAL MIN_CORRECT

set(accuracy_pattern "\\[EVALUATION\\][^\n]*\\[ACCURACY[ ]*([0-9]+) out of[ ]*([0-9]+)\\]")

macro(nn_run result_var output_var)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "NN_TRAIN_CSV=${TRAIN_CSV}"
                "NN_TEST_CSV=${TEST_CSV}"
                "NN_EPOCHS=${EPOCHS}"
                "NN_SEED=${SEED}"
                ${ARGN}
                -- "${NN_EXE}"
        WORKING_DIRECTORY "${WORK_DIR}"
        RESULT_VARIABLE ${result_var}
        OUTPUT_VARIABLE ${output_var}
        ERROR_VARIABLE nn_stderr)
    string(APPEND ${output_var} "${nn_stderr}")
endmacro()

if(CASE STREQUAL "runs")
    nn_run(result output)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "nn exited with ${result}\n${output}")
    endif()

elseif(CASE STREQUAL "loads")
    nn_run(result output)
    if(NOT output MATCHES "out of ${EXPECT_TRAIN}")
        message(FATAL_ERROR "Training set did not report ${EXPECT_TRAIN} samples\n${output}")
    endif()

elseif(CASE STREQUAL "accuracy")
    nn_run(result output)
    if(NOT output MATCHES "${accuracy_pattern}")
        message(FATAL_ERROR "No [EVALUATION] accuracy line in output\n${output}")
    endif()
    set(correct ${CMAKE_MATCH_1})
    set(total ${CMAKE_MATCH_2})
    if(NOT total EQUAL EXPECT_TOTAL)
        message(FATAL_ERROR "Expected ${EXPECT_TOTAL} evaluation samples, saw ${total}")
    endif()
    if(correct LESS MIN_CORRECT)
        message(FATAL_ERROR "Accuracy ${correct} / ${total} is below the ${MIN_CORRECT} floor")
    endif()
    message(STATUS "accuracy ${correct} / ${total}")

elseif(CASE STREQUAL "reproducible")
    nn_run(first_result first_output)
    nn_run(second_result second_output)
    string(REGEX MATCH "${accuracy_pattern}" first_line "${first_output}")
    string(REGEX MATCH "${accuracy_pattern}" second_line "${second_output}")
    if(NOT first_line)
        message(FATAL_ERROR "No [EVALUATION] accuracy line in output\n${first_output}")
    endif()
    if(NOT first_line STREQUAL second_line)
        message(FATAL_ERROR "Seeded runs diverged:\n  ${first_line}\n  ${second_line}")
    endif()

elseif(CASE STREQUAL "rejects-invalid-topology")
    nn_run(result output "NN_HIDDEN1=0")
    if(result EQUAL 0)
        message(FATAL_ERROR "A zero-width hidden layer was accepted\n${output}")
    endif()

else()
    message(FATAL_ERROR "Unknown CASE '${CASE}'")
endif()
