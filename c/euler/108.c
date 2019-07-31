#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>


// count factors where value > 1
uint32_t countFactors(uint32_t value) {
  uint32_t factors = 2;
  uint32_t maxValue = floor(value/2.0);
  for(uint32_t i = 2; i <= maxValue; i++) {
    if(value%i == 0) {
      factors++;
    }
  }
  return factors;
}

int main() {
  // the problem is fundamentally equivalent to how many factors n has
  // knowing that, we can simply solve for all factors of n
  for(uint32_t i = 1; i < 121; i++) {
    uint32_t factors = countFactors(i);
      printf("FOUND n %" PRIu32 " WITH %" PRIu32 " DISTINCT SOLUTIONS\n", i, factors);
    if(factors > 10) {
      printf("FOUND n %" PRIu32 " WITH %" PRIu32 " DISTINCT SOLUTIONS\n", i, factors);
      //return 0;
    }
  }
  return 1;
}

