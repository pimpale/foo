#[derive(Copy, Clone)]
struct Thing {
    field: u32,
}

fn main() -> () {
    let mut buffer = vec![Thing { field: 0 }; 50];

    for i in 0..50 {
        buffer[i] = Thing { field: i as u32 };
    }

    let ok = buffer.iter();
}
