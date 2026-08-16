cmake_minimum_required(VERSION 3.21)

project(nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(nn_x86 ON)
else()
    set(nn_x86 OFF)
endif()

option(NN_ENABLE_AVX2 "Compile the AVX2/FMA kernels" ${nn_x86})

add_executable(nn
    src/dataset.cpp
    src/main.cpp
    src/nn.cpp
    src/pipeline.cpp
    src/trainer.cpp
    src/activation.h
    src/common.h
    src/dataset.h
    src/nn.h
    src/pipeline.h
    src/trainer.h
)

if(MSVC)
    target_compile_definitions(nn PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(nn PRIVATE /W3 /permissive- /fp:fast
        $<$<NOT:$<CONFIG:Debug>>:/O2 /Oi /Ot /GL>)
    target_link_options(nn PRIVATE $<$<NOT:$<CONFIG:Debug>>:/LTCG>)
    if(NN_ENABLE_AVX2)
        target_compile_options(nn PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(nn PRIVATE -Wall -Wextra -ffast-math
        $<$<NOT:$<CONFIG:Debug>>:-O3>)
    if(NN_ENABLE_AVX2)
        target_compile_options(nn PRIVATE -mavx2 -mfma)
    endif()
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
