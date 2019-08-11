#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

int main() {
  while(!feof(stdin)) {
    char hexstring[20];
    fgets(hexstring, 20, stdin);
    printf("%" PRIi64 "\n", strtoll(hexstring, NULL, 16));
  }
}
