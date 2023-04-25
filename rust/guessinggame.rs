struct Foo {
    a: String,
    b: String,
}

fn main() {
    let x = Foo {
        a: String::from("a"),
        b: String::from("b"),
    };

    let q:u8 = 1;

    let xr = &x;

    println!("{}", xr.a);

}
