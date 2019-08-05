import java.util.*;
import java.util.stream.*;

class lambda {
  public static void main(String[] args) {
    ArrayList<Integer> list = new ArrayList(Arrays.asList(new int[] {1, 2, 3, 4, 5}));

    list = list
      .map((x) -> x + 1)
      .collect(Collectors.toCollection(ArrayList::new));
    System.out.println(list);
  }
}
