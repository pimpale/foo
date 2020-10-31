#[derive(Copy, Clone)]



struct Thing {
    field: u32,
}

fn myfn2(a:&Thing) {
  let x = *a;
  println!("{}", a.field);
}


fn main() {
  let foo = Thing { field: 0 };
  let fooref2 = &foo;
  myfn2(fooref2);

  println!("{}", foo.field);
}
