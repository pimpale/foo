#[derive(Copy, Clone)]
struct Thing {
    field: u32,
}

enum Maybe {
  Option1,
  Option2
}

fn main() -> () {
    let mut buffer = vec![Thing { field: 0 }; 50];

    for i in 0..50 {
        buffer[i] = Thing { field: i as u32 };
    }

    let foo = Maybe::Option1;

    let thing = match foo {
      Maybe::Option1 => "cool",
      _ => "nice"
    };

    println!("{}", thing);
}
