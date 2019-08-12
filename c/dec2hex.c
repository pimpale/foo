#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main() {
  char *decstring = NULL;
  size_t bufsize = 0;
  while(getline(&decstring, &bufsize, stdin) != -1) {
    printf("%" PRIx64 "\n", strtoll(decstring, NULL, 16));
  }
  free(decstring);
}
