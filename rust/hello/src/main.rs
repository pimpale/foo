struct Triplet {
  x: i32,
  y: i32,
  z: i32,
}

fn main() {
  let Triplet { x, y, z } ; Triplet { x: 1, y: 2, z: 3 };

  x = 20;
  y = 20;
  z = 20;

  println!("Hello, world! {} {} {}",x,y,z);
}
