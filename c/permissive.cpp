#include <stdio.h>

void test() {
  return 10;
}

int main() {
  puts("test1");
  int oof = 0;
  test();

  printf("%d\n", oof);

  return 0;
}
