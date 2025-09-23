use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cpp_source_dir = PathBuf::from("cpp-wrapper");
    let cpp_build_dir = cpp_source_dir.join("build");

    if !cpp_build_dir.exists() {
        std::fs::create_dir_all(&cpp_build_dir).expect("failed to create build dir");
    }

    println!("cargo:rerun-if-changed=cpp-wrapper/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp-wrapper/embedder_wrapper.h");
    println!("cargo:rerun-if-changed=cpp-wrapper/embedder_wrapper.cpp");

    let libtorch_path = match env::var("LIBTORCH") {
        Ok(path) => path,
        Err(_err) => {
            let homebrew_libtorch_path = PathBuf::from("/opt/homebrew/opt/pytorch");
            if homebrew_libtorch_path.exists() && homebrew_libtorch_path.is_dir() {
                homebrew_libtorch_path.to_str().unwrap().to_string()
            } else {
                panic!(
                    "LIBTORCH environment variable not set, and couldn't find a homebrew installation of pytorch! Please point it to your libtorch installation."
                )
            }
        }
    };

    let cmake_config_status = Command::new("cmake")
        .current_dir(&cpp_build_dir)
        .arg(format!("-DCMAKE_PREFIX_PATH={}", libtorch_path))
        .arg("..")
        .spawn()
        .unwrap()
        .wait()
        .expect("cmake config failed");

    if !cmake_config_status.success() {
        panic!("cmake config failed");
    }

    let cmake_build_status = Command::new("cmake")
        .current_dir(&cpp_build_dir)
        .arg("--build")
        .arg(".")
        .arg("--config")
        .arg("Release")
        .spawn()
        .unwrap()
        .wait()
        .expect("Failed to execute cmake build command");

    if !cmake_build_status.success() {
        panic!("CMake build failed!");
    }

    println!("cargo:rustc-link-search=native={}", cpp_build_dir.display());
    println!("cargo:rustc-link-lib=static=embedder_wrapper");

    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
