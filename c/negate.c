#include <limits.h>
#include <stdio.h>

int main() {
  int x = -(INT_MIN/2);
  int y = INT_MIN/2;

  unsigned int ux = x;
  unsigned int uy = y;

  printf("%i\n", x);
  printf("%i\n", y);

  printf("%u\n", (ux - uy));
  printf("%i\n", -(y - x));

  printf("%i\n", (ux - uy) == -(unsigned)(y -x));
}
