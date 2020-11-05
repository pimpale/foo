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
