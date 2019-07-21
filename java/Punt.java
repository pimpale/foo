class Punt {
  public static void main(String[] args) {
    Punt p  = new Punt();
    Object o = p.go(new Punt());
    Punt[] p2  = (Punt[])o;
    System.out.println(p.equals(p2[0]) + " " + (p == p2[0]));
  }

  Punt[] go(Punt p) {
    Punt[] p2 = {p};
    return p2;
  }
}


