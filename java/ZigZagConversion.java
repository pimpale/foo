import java.util.ArrayList;

public class ZigZagConversion {


// 0123456789
//
// 02468
// 13579
//
// 0 4 8
// 13579
// 2 6  
//
// 0  6
// 1 57
// 24 8
// 3  9
//
// 0   8
// 1  79
// 2 6 A
// 35  B
// 4   C
//


public String convert(String s, int numRows) {
  int centerRows = Math.max(numRows - 2, 0);

  String ret = "";
  // first we add in the Top Row
  // Repeats every numRow + centerRows
  for(int i = 0; i < s.length(); i+=numRows+centerRows) {
    ret += s.charAt(i);
  }

  if(numRows == 1) {
      // 1 is an exceptional case, because there is no bottom row
      return ret;
  }

  for(int row = 1; row <= centerRows; row++) {
    // we need to get 2:
    // The first char at a multiple of (numRows + centerRows) with offset row
    // The second char at a multiple of (numRows + centerRows) with offset
    for(int i = row; i < s.length(); i+=numRows+centerRows) {
      ret += s.charAt(i);
      int i2 = i + (numRows + centerRows) - 2*row;
      if(i2 < s.length()) {
          ret += s.charAt(i2);
      }
    }
  }

  // now do the bottom row
  for(int i = numRows-1; i < s.length(); i+=numRows+centerRows) {
    ret += s.charAt(i);
  }
  return ret;
}


  public static void main(String[] args) {
    System.out.println(convert("0123456789", 3));
  }
}
