interface inter { 
  public static void hello() {
    System.out.println("hello from inter");
  }
}

class clas implements inter{
  public static void hello() {
    System.out.println("hello from clas");
  }
}
class main {

  public static void main(String[] args) {
    clas thing = new clas();
    thing.hello();
  }
}

