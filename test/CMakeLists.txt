# Two runs of the same smoke suite: a generated dataset small enough for CI, and the real
# Fashion-MNIST CSVs when they have actually been pulled from git-lfs.

set(nn_smoke_script "${CMAKE_CURRENT_SOURCE_DIR}/smoke_test.cmake")
set(nn_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

function(nn_add_smoke_suite prefix)
    cmake_parse_arguments(ARG ""
        "TRAIN;TEST;EPOCHS;SEED;EXPECT_TRAIN;EXPECT_TOTAL;MIN_CORRECT;FIXTURE" "LABELS" ${ARGN})

    foreach(case runs loads accuracy reproducible rejects-invalid-topology)
        add_test(NAME ${prefix}.${case}
            COMMAND "${CMAKE_COMMAND}"
                    "-DNN_EXE=$<TARGET_FILE:nn>"
                    "-DWORK_DIR=${CMAKE_SOURCE_DIR}"
                    "-DCASE=${case}"
                    "-DTRAIN_CSV=${ARG_TRAIN}"
                    "-DTEST_CSV=${ARG_TEST}"
                    "-DEPOCHS=${ARG_EPOCHS}"
                    "-DSEED=${ARG_SEED}"
                    "-DEXPECT_TRAIN=${ARG_EXPECT_TRAIN}"
                    "-DEXPECT_TOTAL=${ARG_EXPECT_TOTAL}"
                    "-DMIN_CORRECT=${ARG_MIN_CORRECT}"
                    -P "${nn_smoke_script}")
        set_tests_properties(${prefix}.${case} PROPERTIES LABELS "${ARG_LABELS}")
        if(ARG_FIXTURE)
            set_tests_properties(${prefix}.${case} PROPERTIES FIXTURES_REQUIRED ${ARG_FIXTURE})
        endif()
    endforeach()
endfunction()

add_test(NAME generated.dataset
    COMMAND "${CMAKE_COMMAND}" "-DOUT_DIR=${nn_generated_dir}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_dataset.cmake")
set_tests_properties(generated.dataset PROPERTIES FIXTURES_SETUP generated_dataset LABELS ci)

nn_add_smoke_suite(generated
    TRAIN "${nn_generated_dir}/train.csv"
    TEST "${nn_generated_dir}/test.csv"
    EPOCHS 20
    SEED 12345
    EXPECT_TRAIN 400
    EXPECT_TOTAL 200
    MIN_CORRECT 150
    FIXTURE generated_dataset
    LABELS ci)

# A pointer file is a few hundred bytes; the real CSV is tens of megabytes.
set(nn_train_csv "${CMAKE_SOURCE_DIR}/data/fashion-mnist_train.csv")
set(nn_test_csv "${CMAKE_SOURCE_DIR}/data/fashion-mnist_test.csv")
set(nn_have_dataset OFF)
if(EXISTS "${nn_train_csv}" AND EXISTS "${nn_test_csv}")
    file(SIZE "${nn_train_csv}" nn_train_bytes)
    if(nn_train_bytes GREATER 1000000)
        set(nn_have_dataset ON)
    endif()
endif()

if(nn_have_dataset)
    nn_add_smoke_suite(dataset
        TRAIN "${nn_train_csv}"
        TEST "${nn_test_csv}"
        EPOCHS 1
        SEED 12345
        EXPECT_TRAIN 60000
        EXPECT_TOTAL 10000
        MIN_CORRECT 7500
        LABELS dataset)
else()
    message(STATUS "nn: data/fashion-mnist_*.csv not present, skipping the dataset smoke suite")
endif()
