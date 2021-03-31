#include <stdio.h>

int main() {
  for(int i = 0; i < 10; i++) {
    fprintf(stdout, "stdout%d\n", i);
  }

  for(int i = 0; i < 10; i++) {
    fprintf(stderr, "stderr%d\n", i);
  }
}
