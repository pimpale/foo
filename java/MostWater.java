public class MostWater {

  // very suck solution
  public int maxArea(int[] height) {
    int max = 0;
    for(int i = 0; i < height.length; i++) {
        for(int j = 0; j < i; j++) {
          int width = i - j;
          int height = Math.min(height[i], height[j]);
          int area = height*width;
          if(area > max) {
              max = area;
          }
        }
    }
    return max;
  }

  // better would be to approach from both ends
  // maintain one pointer at the beginning and at the end
  // advance the one currently at the lesser
  // will do later

  public static void main(String[] args) {

  }

}
