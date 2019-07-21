#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

typedef struct __attribute__((__packed__)) {
  void* data;
  uint8_t dummy0;
  uint8_t dummy1;
  uint8_t dummy2;
  uint8_t dummy3;
} DataStructure;


void initDataStructure1(DataStructure* dsp) {
  dsp->dummy0 = 0;
  dsp->dummy1 = 0;
  dsp->dummy2 = 0;
  dsp->dummy3 = 0;
  dsp->data = malloc(13);
  memcpy(dsp->data, "hello1234567", 13);
}

void initDataStructure2(DataStructure* dsp) {
  dsp->dummy0 = 0;
  dsp->dummy1 = 0;
  dsp->dummy2 = 0;
  dsp->dummy3 = 0;

  void* ptr = malloc(6); 
  memcpy(ptr, "hello", 6);

  dsp->data = ptr;
}


void freeDataStructure(DataStructure* dsp) {
  free(dsp->data);
}


int main() {
  DataStructure d1;
  DataStructure d2;
  initDataStructure1(&d1);
  initDataStructure1(&d2);
}
