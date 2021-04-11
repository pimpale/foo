use std::fmt::Binary;

fn print_bin<T:Binary>(x:T) {
  println!("{:#034b}", x);
}

fn xbyte(word:i32, bytenum:i32) -> i32 {
 // ((word >> (bytenum << 3)) << 24) >> 24
  (word << (3 - bytenum << 3)) >> 24
}


fn main() {
  let word:i32 = -1;
  let thing =xbyte(0b011111111, 0);
  print_bin(thing);
  println!("{}", thing);
}
