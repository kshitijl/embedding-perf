fn main() {
    eprintln!("Linking!");
    println!("cargo:rustc-link-search=native=./cpp-wrapper/build");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../../cpp-wrapper/build");
}
