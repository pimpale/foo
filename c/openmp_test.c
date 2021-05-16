#include <omp.h>
#include <stdio.h>

void func(void) { puts("ok"); }

int main() {
#pragma omp parallel
  {
    omp_set_num_threads(2);
#pragma omp for
    for (int i = 0; i < 5; i++) {
      func();
    }
  }
}
