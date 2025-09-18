LIBTORCH_PATH := "/Users/kshitijlauria/Downloads/libtorch"
BUILD_DIR := "contestants/cpp-libtorch-torchscript/build"

default: build

clean:
    rm -rf {{BUILD_DIR}}

configure-cmake profile="Release":
    #!/usr/bin/env bash
    if [ ! -f {{BUILD_DIR}}/Makefile ]; then
        echo "No Makefile found, running configure..."
        mkdir -p {{BUILD_DIR}}
        cd {{BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH={{LIBTORCH_PATH}} -DCMAKE_BUILD_TYPE={{profile}} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    fi

build profile="Release": (configure-cmake profile)
    cd {{BUILD_DIR}} && cmake --build . --config {{profile}}

# Shortcuts for common builds
debug: clean (build "Debug")
release: clean (build "Release")

rebuild profile="Release": clean (build profile)

run profile="Release": (build profile)
    {{BUILD_DIR}}/main generate-torchscript-models/output/all-MiniLM-L6-v2/model.pt data/tokenized/all-MiniLM-L6-v2 contestants/cpp-libtorch-torchscript/embeddings mps
