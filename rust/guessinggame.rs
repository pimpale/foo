use std::io;

fn stuff(x:String) {
    &x
    // x is deallocated
}

fn main() {
    let x = stuff(String::from("hi"));
}
