#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  char hostname[255];
  gethostname(hostname, 255);
  puts(hostname);
}
