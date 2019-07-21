interface Hero {
  default String m1() {
    return "m1";
  }
}

public class Xmen implements Hero {
  @Override
  public String m1() {
    return "m1 over";
  }

  public static void main(String[] args) {
    Xmen n = new Xmen();
    n.go(n);
  }
  void go(Xmen n) {
    System.out.print(n.m1());
    System.out.print(Hero.super.m1());
    System.out.println();
  }
}

