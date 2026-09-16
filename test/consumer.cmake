set(consumer_build "${SCRATCH_DIR}/consumer-build")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${ROOT}/test/consumer" -B "${consumer_build}"
        -G "${GENERATOR}" "-DCMAKE_C_COMPILER=${COMPILER}"
        "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}" "-DCMAKE_BUILD_TYPE=${CONFIGURATION}"
        "-DLIBCMLP_ROOT=${ROOT}" -DBUILD_TESTING=ON
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer configure failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}" --config "${CONFIGURATION}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer build failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    -C "${CONFIGURATION}" --show-only=json-v1
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer test inventory failed: ${output}\n${error}")
endif()
string(JSON count LENGTH "${output}" tests)
string(JSON name GET "${output}" tests 0 name)
if(NOT count EQUAL 1 OR NOT name STREQUAL "consumer.smoke")
    message(FATAL_ERROR "Library polluted consumer CTest registration: ${output}")
endif()
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    -C "${CONFIGURATION}" --output-on-failure
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer test failed: ${output}\n${error}")
endif()
message(STATUS "Embedded C library builds and passes with exactly one consumer test")
