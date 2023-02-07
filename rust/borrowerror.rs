fn main() {
  let mut x = String::from("hello");
  let y = &mut x;
  let z = &mut x;
  print!("{}", z);
  print!("{}", y);
}
