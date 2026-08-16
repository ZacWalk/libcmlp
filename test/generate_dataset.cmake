# Writes a small, deterministic Fashion-MNIST-shaped CSV pair so the smoke suite can run
# without the real dataset, which lives in git-lfs and is far too large for CI to pull.
#
#   cmake -DOUT_DIR=<dir> -P generate_dataset.cmake

if(NOT OUT_DIR)
    message(FATAL_ERROR "OUT_DIR is required")
endif()

set(classes 10)
set(features 32)
set(train_rows 400)
set(test_rows 200)

file(MAKE_DIRECTORY "${OUT_DIR}")

set(rng_state 12345)

macro(next_random out upper)
    math(EXPR rng_state "(${rng_state} * 1103515245 + 12345) & 0x7fffffff")
    math(EXPR ${out} "${rng_state} % ${upper}")
endmacro()

set(header "label")
foreach(pixel RANGE 1 ${features})
    string(APPEND header ",pixel${pixel}")
endforeach()

# Each class is a flat block of pixels around its own mean, buried in noise wide enough
# to overlap its neighbours. Learnable, but only if the forward and backward passes work.
macro(write_dataset path row_count)
    set(lines "${header}")
    foreach(row RANGE 1 ${row_count})
        next_random(label ${classes})
        math(EXPR base "16 + ${label} * 22")
        set(line "${label}")
        foreach(pixel RANGE 1 ${features})
            next_random(noise 61)
            math(EXPR value "${base} + ${noise} - 30")
            if(value LESS 0)
                set(value 0)
            elseif(value GREATER 255)
                set(value 255)
            endif()
            string(APPEND line ",${value}")
        endforeach()
        list(APPEND lines "${line}")
    endforeach()
    string(REPLACE ";" "\n" text "${lines}")
    file(WRITE "${path}" "${text}\n")
endmacro()

write_dataset("${OUT_DIR}/train.csv" ${train_rows})
write_dataset("${OUT_DIR}/test.csv" ${test_rows})
