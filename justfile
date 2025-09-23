CPP_LIBTORCH_TS_BUILD_DIR := "contestants/cpp-libtorch-ts/build"
CPP_LIBTORCH_TS_LIBTORCH_PATH := "/opt/homebrew/opt/pytorch"
RUST_CPP_LIBTORCH_TS_DIR := "contestants/rust-cpp-libtorch-ts"
RUST_CPP_LIBTORCH_TS_BUILD_DIR := RUST_CPP_LIBTORCH_TS_DIR + "/cpp-wrapper/build"
RUST_TCH_TS_DIR := "contestants/rust-tch-ts"
RUST_TCH_TS_LIBTORCH_DIR := "~/Downloads/libtorch-2.7.0"
RUST_CANDLE_DIR := "contestants/rust-candle"

default: evaluate

build-results:
    pushd results; ./build.sh

deploy-website: 
    pushd website; yarn build; yarn deploy

evaluate:
    uv run --directory results evaluate-correctness.py

generate_reference_data:
    uv run --directory data tokenize-and-reference-embeddings.py

rebuild-all: clean-all build-all

clean-all: clean-cpp-libtorch-ts clean-rust-cpp-libtorch-ts clean-rust-tch-ts clean-rust-candle
build-all: typecheck-python build-cpp-libtorch-ts build-rust-cpp-libtorch-ts build-rust-tch-ts build-rust-candle

typecheck-python:
    uvx ty check

# rust-candle

clean-rust-candle:
    cd {{RUST_CANDLE_DIR}} && cargo clean

build-rust-candle:
    cd {{RUST_CANDLE_DIR}} && cargo build --release    

# rust-tch-ts

clean-rust-tch-ts:
    cd {{RUST_TCH_TS_DIR}} && cargo clean

build-rust-tch-ts:
    cd {{RUST_TCH_TS_DIR}} && LIBTORCH={{RUST_TCH_TS_LIBTORCH_DIR}} cargo build --release    

# rust-cpp-libtorch-ts

clean-rust-cpp-libtorch-ts:
    rm -rf {{RUST_CPP_LIBTORCH_TS_BUILD_DIR}}
    cd {{RUST_CPP_LIBTORCH_TS_DIR}} && cargo clean

build-rust-cpp-libtorch-ts:
    cd {{RUST_CPP_LIBTORCH_TS_DIR}} && cargo build --release

# cpp-libtorch-ts commands

clean-cpp-libtorch-ts:
    rm -rf {{CPP_LIBTORCH_TS_BUILD_DIR}}

configure-cmake-cpp-libtorch-ts profile="Release":
    #!/usr/bin/env bash
    if [ ! -f {{CPP_LIBTORCH_TS_BUILD_DIR}}/Makefile ]; then
        echo "No Makefile found, running configure..."
        mkdir -p {{CPP_LIBTORCH_TS_BUILD_DIR}}
        cd {{CPP_LIBTORCH_TS_BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH={{CPP_LIBTORCH_TS_LIBTORCH_PATH}} -DCMAKE_BUILD_TYPE={{profile}} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    fi

build-cpp-libtorch-ts profile="Release": (configure-cmake-cpp-libtorch-ts profile)
    cmake --build {{CPP_LIBTORCH_TS_BUILD_DIR}} --config {{profile}}

