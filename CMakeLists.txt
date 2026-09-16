cmake_minimum_required(VERSION 3.24)

project(libcmlp VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(nn_x86 ON)
else()
    set(nn_x86 OFF)
endif()

option(NN_ENABLE_AVX2 "Compile the AVX2/FMA kernels" ${nn_x86})
option(NN_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(CMLP_BUILD_EXAMPLE "Build the Fashion-MNIST example CLI" ${PROJECT_IS_TOP_LEVEL})

if(NN_ENABLE_AVX2 AND NOT nn_x86)
    message(FATAL_ERROR "NN_ENABLE_AVX2 requires an x86 target; use -DNN_ENABLE_AVX2=OFF")
endif()

include(cmake/compiler-options.cmake)

add_library(cmlp STATIC
    include/cmlp.h
    src/nn.c
    src/activation.h
    src/common.h
    src/random.h
)
target_include_directories(cmlp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(cmlp PUBLIC c_std_11)
cmlp_configure_target(cmlp)

if(NN_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(cmlp PRIVATE /arch:AVX2)
    else()
        target_compile_options(cmlp PRIVATE -mavx2 -mfma)
    endif()
endif()
if(WIN32)
    target_link_libraries(cmlp PUBLIC bcrypt)
else()
    target_link_libraries(cmlp PUBLIC m)
endif()

if(CMLP_BUILD_EXAMPLE)
    add_executable(mlp-cli
        src/main.c
        src/dataset.c
        src/pipeline.c
        src/trainer.c
        src/dataset.h
        src/pipeline.h
        src/trainer.h
    )
    target_link_libraries(mlp-cli PRIVATE cmlp)
    cmlp_configure_target(mlp-cli)
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if(BUILD_TESTING)
        if(NOT CMLP_BUILD_EXAMPLE)
            message(FATAL_ERROR "Standalone tests require CMLP_BUILD_EXAMPLE=ON")
        endif()
        add_subdirectory(test)
    endif()
endif()
