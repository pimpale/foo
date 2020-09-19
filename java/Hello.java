class Hello {

  static void reverse(char[] in) {
		for(int i = 0; i < in.length/2; i++) {
			char tmp = in[i];
			in[i] = in[in.length-1-i];
			in[in.length-1-i] = tmp;
		}
  }

  static int foo(int in) {
    if(input == Integer.MIN_VALUE) {
        return 0;
    }

    boolean prefix = "";
    if(in < 0) {
        prefix = "-";
        in = -in;
    } 

    char[] inarr = Integer(in).toString().toCharArray();
    reverse(inarr);
    long ret = Long.parseLong(new String(inarr));

    if(ret > Integer.MAX_VALUE) {
      return 0;
    }
    return (int)ret;
  }


  public static void main(String[] args) {
    System.out.println("Hello");

    int input = 100;


  }
}
