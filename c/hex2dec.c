#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main() {
  char *hexstring = NULL;
  size_t bufsize = 0;
  while(getline(&hexstring, &bufsize, stdin) != -1) {
    printf("%" PRIi64 "\n", strtoll(hexstring, NULL, 16));
  }
  free(hexstring);
}
