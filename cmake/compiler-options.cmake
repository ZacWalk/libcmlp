function(cmlp_configure_target target)
    if(MSVC)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4 /fp:precise
            "$<$<CONFIG:Release>:/O2;/Oi;/Ot>")
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic
            "$<$<CONFIG:Release>:-O3>")
    endif()

    if(NN_SANITIZE)
        if(MSVC)
            message(FATAL_ERROR "NN_SANITIZE requires GCC or Clang with ASan/UBSan; use the Linux sanitizer preset")
        elseif(CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined
                -fno-sanitize-recover=all -fno-omit-frame-pointer)
            # PUBLIC propagates the runtime when a consumer links the instrumented archive.
            target_link_options(${target} PUBLIC -fsanitize=address,undefined
                -fno-sanitize-recover=all)
        else()
            message(FATAL_ERROR "NN_SANITIZE is not supported by ${CMAKE_C_COMPILER_ID}")
        endif()
    endif()
endfunction()
