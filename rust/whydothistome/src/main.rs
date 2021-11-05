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


      struct Sub {
          x: String
      }

      struct Super {
          a: Sub,
          b: Sub
      }

      let mut ok = Super {
          a: Sub { x: String::new() },
          b: Sub { x: String::new() }
      };

      let borrow_a = &mut ok.a;
      let borrow_b = &mut ok.b;

      //let borrow2 = &mut ok;


      //borrow2.a.x = String::from("hi");
      //borrow2.b.x = String::from("hi");

      borrow_a.x = String::from("tho");
      borrow_b.x = String::from("tho");



      let aa = 2;
      let bb = 2;
      (aa, bb) = (1,2);


}
