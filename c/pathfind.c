#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#define xsize 20
#define ysize 13

char charmap[ysize][xsize] = {
  "XXXXXXXXXXXXXXXXXXX",
  "XXXXXXXXXXXXXXXXXXX",
  "XOOOOOOXOOOOXXXXXXX",
  "XOXXXXXXXXXOXXXXOOO",
  "XOOOOXOOOOOOOOOXOXX",
  "XXXXOXOXXXXOXXOXOXX",
  "XOXXOXOOOOXOXXOXOXX",
  "XOXXOXXXXOXOXXOXOXX",
  "XOOOOOOXXOXOXXOXOOX",
  "XXXXOXOXXOXXXXOXOXX",
  "OOOOOXOXXOXXXXOXOXX",
  "XXXXXXOOOOXXXXOOOXX",
  "XXXXXXXXXXXXXXXXXXX"
};

uint32_t genmap[ysize][xsize];
uint32_t bakmap[ysize][xsize];


void step() {
  // increase generation
  for(uint32_t x = 0; x < xsize; x++) {
    for(uint32_t y = 0; y < ysize; y++) {
      if(genmap[y][x] != 0) {
        genmap[y][x]++;
      }
    }
  }

  for(int32_t x = 0; x < xsize; x++) {
    for(int32_t y = 0; y < ysize; y++) {
      // for all the just promoted generation
      if(genmap[y][x] == 2) {

        // test each one
        for(int32_t rx = x -1; rx <= x+1; rx++) {
          for(int32_t ry = y -1; ry <= y+1; ry++) {
            // if it is valid
            if(rx >= 0 && rx < xsize && ry >= 0 && ry < ysize) {
              // if open and the map is free
              if(genmap[ry][rx] != 0) {
                continue;
              } else if(charmap[ry][rx] == 'E') {
                // victory
                backtrack(rx, ry);
              } else if(charmap[ry][rx] == 'O') {
                genmap[ry][rx] = 1;
              }
            }
          }
        }
      }
    }
  }
}

void backtrack(int32_t x, int32_t y) {
  while(1) {
    bakmap[y][x] = 1;
    int currentmapval = genmap[y][x];
    for(int32_t rx = x -1; rx <= x+1; rx++) {
      for(int32_t ry = y -1; ry <= y+1; ry++) {
        // if it is valid
        if(rx >= 0 && rx < xsize && ry >= 0 && ry < ysize) {
          if(bakmap
      }



int main() {
  // preprocess map
  for(uint32_t y = 0; y < ysize; y++) {
    if(charmap[y][0] == 'O') {
      charmap[y][0] = 'S';
      genmap[y][0] = 1;
    }
    if(charmap[y][xsize-1] == 'O') {
      charmap[y][xsize-1] = 'E';
    }
  }


  // step
  while(1) {
    step();
    for(uint32_t y = 0; y < ysize; y++) {
      for(uint32_t x = 0; x < xsize; x++) {
        printf("%c", genmap[y][x] + '0');
      }
      printf("\n");
    }
    printf("\n\n");
  }
}


