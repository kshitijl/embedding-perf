cmake_minimum_required(VERSION 3.18)
project(embedder_wrapper)

# Generate include dirs so the LSP clangd can find include dirs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ standard required by PyTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Build type (Release for optimized performance)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create shared library
add_library(embedder_wrapper SHARED 
    embedder_wrapper.cpp
)

# Include directories
target_include_directories(embedder_wrapper PRIVATE 
    ${TORCH_INCLUDE_DIRS}
)

# Link against PyTorch libraries
target_link_libraries(embedder_wrapper 
    ${TORCH_LIBRARIES}
)

# Set properties for the shared library
set_target_properties(embedder_wrapper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN ON
)

# Platform-specific settings
if(APPLE)
    set_target_properties(embedder_wrapper PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(embedder_wrapper PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Copy torch libraries to build directory (optional, helps with deployment)
if(MSVC)
    # Windows
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET embedder_wrapper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:embedder_wrapper>)
endif()

# Print configuration info
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

# Optional: Installation rules
install(TARGETS embedder_wrapper
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES embedder_wrapper.h
    DESTINATION include
)
