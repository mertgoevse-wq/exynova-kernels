cmake_minimum_required(VERSION 3.22)
project(exynova-kernels VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EXYNOVA_BUILD_EXAMPLES "Build game_npc_ai_demo and vulkan_probe executables" ON)
option(EXYNOVA_BUILD_SHARED "Build exynova_kernels_shared dynamic library" ON)

set(EXYNOVA_KERNELS_SOURCES
    src/cpu_ref.cpp
    src/vulkan_probe.cpp
    src/vulkan_backend.cpp
    src/llm_engine.cpp
    src/enn_backend.cpp
    src/ir.cpp
    src/partitioner.cpp
    src/artifact.cpp
    src/nnc_converter.cpp
    src/exynova_sdk.cpp
)

# Static library (Host + Android ARM64 NDK)
add_library(exynova_kernels STATIC
    ${EXYNOVA_KERNELS_SOURCES}
)
target_include_directories(exynova_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(exynova_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(ANDROID)
    target_compile_definitions(exynova_kernels PUBLIC
        EXYNOVA_ANDROID
        VK_USE_PLATFORM_ANDROID_KHR
    )
    target_link_libraries(exynova_kernels PUBLIC
        vulkan
        dl
        log
        android
    )
endif()

# Shared library (for Unity Native Plugin / Unreal NDK / Godot GDExtension / JNI)
if(EXYNOVA_BUILD_SHARED)
    add_library(exynova_kernels_shared SHARED
        ${EXYNOVA_KERNELS_SOURCES}
    )
    target_include_directories(exynova_kernels_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(exynova_kernels_shared PRIVATE
        EXYNOVA_SHARED_EXPORTS
    )
    set_target_properties(exynova_kernels_shared PROPERTIES
        OUTPUT_NAME "exynova_kernels"
        ARCHIVE_OUTPUT_NAME "exynova_kernels_shared"
    )
    if(ANDROID)
        target_compile_definitions(exynova_kernels_shared PUBLIC
            EXYNOVA_ANDROID
            VK_USE_PLATFORM_ANDROID_KHR
        )
        target_link_libraries(exynova_kernels_shared PUBLIC
            vulkan
            dl
            log
            android
        )
    endif()
endif()

if(EXYNOVA_BUILD_EXAMPLES)
    enable_testing()

    add_executable(game_npc_ai_demo
        examples/game_npc_ai_demo.cpp
    )
    target_link_libraries(game_npc_ai_demo PRIVATE exynova_kernels)
    add_test(NAME game_npc_ai_demo COMMAND game_npc_ai_demo)

    add_executable(vulkan_probe
        examples/vulkan_probe.cpp
    )
    target_link_libraries(vulkan_probe PRIVATE exynova_kernels)
    add_test(NAME vulkan_probe COMMAND vulkan_probe)

    add_executable(flappy_bird_ai_bench
        examples/flappy_bird_ai/flappy_bird_ai_bench.cpp
    )
    target_link_libraries(flappy_bird_ai_bench PRIVATE exynova_kernels)
    add_test(NAME flappy_bird_ai_bench COMMAND flappy_bird_ai_bench)
endif()
