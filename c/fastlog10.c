#include <stdint.h>
#include <stdio.h>

uint64_t mylog2(uint64_t x) {
   return __builtin_clz(x) ^ 31;
}

int main() {
  // LN(10) / LN(2)
  const double log10const = 3.32192809488;
  mylog2(100);
}
