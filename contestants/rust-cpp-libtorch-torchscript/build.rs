fn main() {
    eprintln!("Linking!");
    println!("cargo:rustc-link-search=native=./cpp-wrapper/build");
    println!("cargo:rustc-link-lib=static=embedder_wrapper");

    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/pytorch/lib");
    // println!("cargo:rustc-link-search=native=/Users/kshitijlauria/Downloads/libtorch/lib");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    println!("cargo:rustc-link-lib=dylib=c++");
}
