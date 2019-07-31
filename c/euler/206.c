#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <math.h>


uint64_t place_arr[] = {
  1l,
  10l,
  100l,
  1000l,
  10000l,
  100000l,
  1000000l,
  10000000l,
  100000000l,
  1000000000l,
  10000000000l,
  100000000000l,
  1000000000000l,
  10000000000000l,
  100000000000000l,
  1000000000000000l,
  10000000000000000l,
  100000000000000000l,
  1000000000000000000l
};


#define PLACEMATCHES(x, placeval, place) ((x/place_arr[place]) % 10 == placeval)

static bool fits(uint64_t val) {
  //1_2_3_4_5_6_7_8_9_0
  if(
      PLACEMATCHES(val, 0, 0) &&
      PLACEMATCHES(val, 9, 2) &&
      PLACEMATCHES(val, 8, 4) &&
      PLACEMATCHES(val, 7, 6) &&
      PLACEMATCHES(val, 6, 8) &&
      PLACEMATCHES(val, 5, 10) &&
      PLACEMATCHES(val, 4, 12) &&
      PLACEMATCHES(val, 3, 14) &&
      PLACEMATCHES(val, 2, 16) &&
      PLACEMATCHES(val, 1, 18)) {
    return true;
  } else {
    return false;
  }
}


int main() {
  uint64_t startint = sqrtl(1020304050607080900l);
  uint64_t endint = sqrtl(1929394959697989990l);

  for(uint64_t i = endint; i > startint; i--) {
    uint64_t square = i*i;
    if(fits(square)) {
      printf("FOUND THE INTEGER %" PRIu64 " WITH SQUARE % " PRIu64 "\n", i, square);
      return 0;
    }

    if(i % 100000 == 0) {
      printf("%" PRIu64 "\n", square);
    }
  }
  printf("COULD NOT FIND INT, PROGRAM INCORRECT\n");
  return 1;
}
