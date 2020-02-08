#include <stdio.h>

int main() {
  printf("%d\n", __LINE__);
  puts(__FILE__);
  puts(__func__);
}
