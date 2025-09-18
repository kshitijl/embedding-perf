cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build type specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -DNDEBUG")
endif()

include(FetchContent)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Generate include dirs so the LSP clangd can find include dirs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch nlohmann_json
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2
)

# Make the content available
FetchContent_MakeAvailable(nlohmann_json)

add_executable(main main.cpp)
target_compile_options(main PRIVATE -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion)
target_link_libraries(main
  "${TORCH_LIBRARIES}"
  nlohmann_json::nlohmann_json
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)
