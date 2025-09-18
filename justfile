LIBTORCH_PATH := "/Users/kshitijlauria/Downloads/libtorch"
BUILD_DIR := "contestants/cpp-libtorch-torchscript/build"

default: build

clean:
    rm -rf {{BUILD_DIR}}

configure-cmake:
    #!/usr/bin/env bash
    if [ ! -f {{BUILD_DIR}}/Makefile ]; then
        echo "No Makefile found, running configure..."
        mkdir -p {{BUILD_DIR}}
        cd {{BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH={{LIBTORCH_PATH}} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    fi

build: configure-cmake
    cd {{BUILD_DIR}} && cmake --build . --config Release

rebuild: clean build

run: build
    head -n1 data/tokenized/all-MiniLM-L6-v2 > /tmp/one-sentence
    {{BUILD_DIR}}/main generate-torchscript-models/output/all-MiniLM-L6-v2/model.pt /tmp/one-sentence contestants/cpp-libtorch-torchscript/embeddings mps 
