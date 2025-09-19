use std::os::raw::c_long;

#[link(name = "embedder_wrapper")]
unsafe extern "C" {
    fn embedder_add_numbers(a: c_long, b: c_long) -> c_long;
}

fn main() {
    let answer = unsafe { embedder_add_numbers(1, 1) };
    println!("Did it work? 1 + 1 = {}", answer);
}
