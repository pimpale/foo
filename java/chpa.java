class Parent {
  public String ok = "hello";

}

class Child extends Parent {
  public String ok = "hi";

  public void go() {
    System.out.println(ok);
  }

  public static void main(String[] args) {
    Child c = new Child();
    c.go();
  }
}

