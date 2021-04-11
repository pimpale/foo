#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

bool test(int ix, int iy) {
  int8_t x = ix;
  int8_t y = iy;

  printf(PRIi8 "\n" PRIi8 "\n\n", x, -x);

  //return ~x + ~y + 1 == ~(x + y);
  return (x < y) == (-x > -y);
}

int main() {

      if (test(-128, -127)) {
          puts("nice");
      }


  // for (int ix = -128; ix <= 127; ix++) {
  //   for (int iy = -128; iy <= 127; iy++) {
  //     if (!test(ix, iy)) {
  //       printf("%i\n%i\n\n", ix, iy);
  //     }
  //   }
  // }
}
