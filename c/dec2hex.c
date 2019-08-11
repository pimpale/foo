#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

int main() {
  while(!feof(stdin)) {
    char hexstring[64];
    fgets(hexstring, 64, stdin);
    printf("%" PRIx64 "\n", strtoll(hexstring, NULL, 10));
  }
}
