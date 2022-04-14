trait Foo {
  fn printField(self);
}

#[derive(Debug)]
struct Cool {
  field: u32,
}

struct Thing {
  field: Cool,
}

impl Foo for Thing {
  fn printField(self) {
    dbg!(self.field);
  }
}

fn myfn2(a: &Thing) {
  let x = &*a;
  dbg!(x.field.field);
}

fn main() {
  let foo = Thing {
    field: Cool { field: 0 },
  };
  let fooref2 = &foo;
  myfn2(fooref2);

}

fn ok0() {
  let mut y = String::from("hi");

  let x = &mut y;

  let mut q = || {
      let y = &mut *x;
      y.push_str("ok");
  };

  q();

  x.push_str("ok");

}

fn ok1() {
  let mut y = String::from("hi");

  let mut w = &mut y;
  let x = &mut w;

  let mut q = || {
      let y = &mut **x;
      y.push_str("ok");
  };

  q();

  (*x).push_str("ok");

}
