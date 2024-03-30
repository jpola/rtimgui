project(HIPRT-TraceKernels)
set(target HIPRT-TraceKernels)

set(GPU_ARCH "gfx1100")
message(STATUS "HIPRT ${HIPRT_BINDIR}")
set(sources trace.cpp)

foreach(source IN LISTS sources)
    string(REPLACE ".cpp" "" target_name ${source})
    add_custom_target(${target_name})

    add_custom_command(TARGET ${target_name} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/$<CONFIG>/)

    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${HIP_PATH}/bin/clang.exe
        ARGS -x hip -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true -O3 --offload-device-only --offload-arch=${GPU_ARCH} --cuda-device-only -fgpu-rdc -emit-llvm --gpu-bundle-output -ffast-math -std=c++17 -c ${CMAKE_CURRENT_SOURCE_DIR}/${source} -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_PRINTF=1 -I ./ -I ${HIP_PATH}/include -I ${HIPRT_PATH}  -o ${target_name}.bc
        COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/${source}"
    )

    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${HIP_PATH}/bin/clang.exe
        ARGS -fgpu-rdc --hip-link --cuda-device-only --offload-arch=${GPU_ARCH}  ${target_name}.bc ${HIPRT_BINDIR}/hiprt02003_5.7_amd_lib_win.bc -o ${target_name}.hipfb
        COMMENT "Linking ${target_name}.bc"
    )

    add_custom_command(TARGET ${target_name} POST_BUILD
               COMMAND ${CMAKE_COMMAND} -E copy_if_different ${target_name}.hipfb ${CMAKE_BINARY_DIR}/$<CONFIG>/)
endforeach()


add_custom_target(HIPRT-TraceKernels)

foreach(source IN LISTS sources)
    string(REPLACE ".cpp" "" target_name ${source})
    add_dependencies(HIPRT-TraceKernels ${target_name})
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/${target_name}.cpp
        PROPERTIES
        HEADER_FILE_ONLY TRUE
    )
    target_sources(HIPRT-TraceKernels PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/${target_name}.cpp
        ${HIP_PATH}/include
        ${HIPRT_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/shared.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Math.h
    )
    
endforeach()