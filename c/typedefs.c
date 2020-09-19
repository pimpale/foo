
struct Foo {
  int member;
};

typedef struct Foo Blah;

typedef struct Foo Thing;

void example1(struct Foo f) {

}

void example2(Blah b) {

}

void example3(Thing t) {

}

int main() {
  struct Foo st;
  Blah b;
  Thing t;

  example1(st);
  example1(b);
  example1(t);


  example2(st);
  example2(b);
  example2(t);
}
