# Required: NN_EXE WORK_DIR SCRATCH_DIR CASE.
# Environment and arguments are separate so negative-path checks exercise the CLI itself.
set(accuracy_pattern "\\[EVALUATION\\][^\n]*\\[ACCURACY[ ]*([0-9]+) out of[ ]*([0-9]+)\\]")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

function(nn_run result_var output_var)
    cmake_parse_arguments(ARG "" "" "ENV;ARGS" ${ARGN})
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                --unset=NN_HIDDEN1 --unset=NN_HIDDEN2 --unset=NN_LR
                --unset=NN_LR_DECAY --unset=NN_MOMENTUM --unset=NN_BATCH_SIZE
                "NN_TRAIN_CSV=${TRAIN_CSV}"
                "NN_TEST_CSV=${TEST_CSV}"
                "NN_EPOCHS=${EPOCHS}"
                "NN_SEED=${SEED}"
                ${ARG_ENV}
                -- "${NN_EXE}" ${ARG_ARGS}
        WORKING_DIRECTORY "${WORK_DIR}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE stderr
        TIMEOUT 100)
    set(${result_var} "${result}" PARENT_SCOPE)
    set(${output_var} "${output}${stderr}" PARENT_SCOPE)
endfunction()

function(nn_require_success result output)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "Process exited with ${result}\n${output}")
    endif()
    if("${output}" MATCHES "\\[LOSS [-+]?([Ii][Nn][Ff]|[Nn][Aa][Nn])")
        message(FATAL_ERROR "Successful process reported nonfinite loss\n${output}")
    endif()
endfunction()

function(nn_require_failure result output)
    if(NOT "${result}" MATCHES "^[1-9][0-9]*$" OR NOT "${output}" MATCHES "Error:")
        message(FATAL_ERROR "Expected a normal nonzero CLI error, got ${result}\n${output}")
    endif()
endfunction()

function(nn_subcheck)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${SCRATCH_DIR}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE stderr
        TIMEOUT 100)
    nn_require_success("${result}" "${output}${stderr}")
    message(STATUS "${output}")
endfunction()

if(CASE STREQUAL "runs")
    nn_run(result output)
    nn_require_success("${result}" "${output}")
    if(RUN_SUBCHECKS)
        nn_subcheck("${CMLP_TEST_EXE}")
        nn_subcheck("${CMAKE_COMMAND}" "-DROOT=${WORK_DIR}" "-DSCRATCH_DIR=${SCRATCH_DIR}"
            "-DGENERATOR=${CONSUMER_GENERATOR}" "-DCOMPILER=${CONSUMER_COMPILER}"
            "-DMAKE_PROGRAM=${CONSUMER_MAKE_PROGRAM}" "-DCONFIGURATION=${CONSUMER_CONFIGURATION}"
            -P "${WORK_DIR}/test/consumer.cmake")
        if(DD_PWSH)
            nn_subcheck("${DD_PWSH}" -NoProfile -File "${WORK_DIR}/test/dd-adoption.ps1"
                -BuildDir "${BUILD_DIR}")
        else()
            nn_subcheck("${CMAKE_COMMAND}" "-DROOT=${WORK_DIR}" "-DBUILD_DIR=${BUILD_DIR}"
                -P "${WORK_DIR}/test/dd-adoption.cmake")
            message(STATUS "PowerShell unavailable: dynamic dd checks skipped; vendor hashes verified")
        endif()
        nn_run(result output ENV "NN_TRAIN_CSV=${SCRATCH_DIR}/missing.csv"
            "NN_TEST_CSV=${SCRATCH_DIR}/missing.csv" ARGS --help)
        nn_require_success("${result}" "${output}")
        if(NOT output MATCHES "Usage: mlp-cli")
            message(FATAL_ERROR "Missing CLI help text\n${output}")
        endif()
    endif()

elseif(CASE STREQUAL "loads")
    if(RUN_SUBCHECKS)
        nn_subcheck("${PIPELINE_TEST_EXE}")
    endif()
    nn_run(result output)
    nn_require_success("${result}" "${output}")
    if(NOT output MATCHES "out of ${EXPECT_TRAIN}")
        message(FATAL_ERROR "Training set did not report ${EXPECT_TRAIN} samples\n${output}")
    endif()
elseif(CASE STREQUAL "accuracy")
    nn_run(result output)
    nn_require_success("${result}" "${output}")
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
    nn_require_success("${first_result}" "${first_output}")
    nn_run(second_result second_output)
    nn_require_success("${second_result}" "${second_output}")
    string(REGEX MATCH "${accuracy_pattern}" first_line "${first_output}")
    string(REGEX MATCH "${accuracy_pattern}" second_line "${second_output}")
    if(NOT first_line OR NOT second_line)
        message(FATAL_ERROR "Missing [EVALUATION] accuracy line\n${first_output}\n${second_output}")
    endif()
    if(NOT first_line STREQUAL second_line)
        message(FATAL_ERROR "Seeded runs diverged:\n  ${first_line}\n  ${second_line}")
    endif()

elseif(CASE STREQUAL "rejects-invalid-topology")
    nn_run(result output ENV "NN_HIDDEN1=0")
    nn_require_failure("${result}" "${output}")
    if(RUN_SUBCHECKS)
        foreach(invalid
            "NN_HIDDEN1=bad" "NN_HIDDEN1=2147483648" "NN_HIDDEN1=" "NN_HIDDEN2=-1"
            "NN_EPOCHS=0" "NN_EPOCHS=-1" "NN_BATCH_SIZE=0" "NN_BATCH_SIZE=bad"
            "NN_LR=0" "NN_LR=-0.1" "NN_LR=nan" "NN_LR=inf" "NN_LR=1e99"
            "NN_LR=1e-99" "NN_LR=0.1garbage" "NN_LR="
            "NN_LR_DECAY=0" "NN_LR_DECAY=nan" "NN_MOMENTUM=1" "NN_MOMENTUM=-1"
            "NN_MOMENTUM=nan" "NN_SEED=-1" "NN_SEED=4294967296" "NN_SEED=1foo" "NN_SEED=")
            nn_run(result output ENV "${invalid}")
            nn_require_failure("${result}" "${output}")
        endforeach()
        nn_run(result output ARGS --unknown)
        nn_require_failure("${result}" "${output}")
        nn_run(result output ARGS --help unexpected)
        nn_require_failure("${result}" "${output}")
        nn_run(result output ENV "NN_TRAIN_CSV=${SCRATCH_DIR}/missing.csv")
        nn_require_failure("${result}" "${output}")
        nn_run(result output ENV "NN_TEST_CSV=${SCRATCH_DIR}/missing.csv")
        nn_require_failure("${result}" "${output}")
    endif()

else()
    message(FATAL_ERROR "Unknown CASE '${CASE}'")
endif()
