#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

char* helpstring = 
"Usage: sethostname HOSTNAME\n"
"Sets hostname\n";

int main(size_t argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, helpstring);
    exit(1);
  }

  char* hostname = argv[1];
  int err = sethostname(hostname, strlen(hostname));
  if(err != 0) {
    perror("sethostname");
    exit(1);
  }
  exit(0);
}
