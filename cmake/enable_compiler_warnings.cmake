# enable_compiler_warnings(<target>)
#
# Attaches the project's warning set (warnings are errors) and -O3 to <target>
# as PRIVATE options, so they apply only to our own sources and never leak to
# consumers of the installed stPS package.
#
# The strict set is C++-only ($<COMPILE_LANGUAGE:CXX>): .cu files are compiled
# by nvcc, whose generated host stubs (old-style casts, 0 as null pointer, ...)
# would trip most of these flags. CUDA sources get a moderate host-side set
# forwarded to the host compiler via -Xcompiler instead.
function(enable_compiler_warnings tgt)
    set(cxx_warnings
        -Wall -Wextra -Werror -Wfatal-errors -Wpedantic -Wconversion
        -Wformat=2 -Wformat-security -Wvla -Wshadow -Wold-style-cast
        -Woverloaded-virtual -Wnon-virtual-dtor -Wzero-as-null-pointer-constant
        -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference
        -Wuseless-cast -Wcast-qual -Wmissing-declarations -Wmisleading-indentation
        -Winit-self -Wsign-conversion -Wfloat-equal -Wdouble-promotion
        -Wswitch-enum -Wundef -Wmaybe-uninitialized -Wpacked -Wstrict-aliasing=3
    )
    set(cuda_warnings
        -Werror=all-warnings
        -Xcompiler=-Wall,-Wextra,-Wshadow,-Werror
    )
    target_compile_options(${tgt} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${cxx_warnings}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_warnings}>"
        # -O3 for every non-Debug config (Release already defaults to -O3;
        # this also lifts RelWithDebInfo/MinSizeRel). Debug keeps -O0 -g.
        "$<$<NOT:$<CONFIG:Debug>>:-O3>"
    )
endfunction()
