#include <stdio.h>
#include <stdint.h>


#define xsize 4
#define ysize 4

char** charmap = {
  "XXXX",
  "OOOO",
  "XXXX"
};

uint32_t mapgens[xsize][ysize];


void step() {

  // increase generation
  for(uint32_t x = 0; x < xsize; x++) {
    for(uint32_t y = 0; y < ysize; y++) {
      if(mapgens[x][y] != 0) {
        mapgens[x][y]++;
      }
    }
  }

  for(uint32_t x = 0; x < xsize; x++) {
    for(uint32_t y = 0; y < ysize; y++) {
      if(mapgens[x][y] != 2) {
        // test each one
        for(uint32_t rx = x -1; rx <= x+1; rx++) {
          for(uint32_t ry = y -1; ry <= y+1; ry++) {
            // if it is valid
            if(rx > 0 && rx < xsize && ry > 0 && ry < ysize) {
              // if open and the map is free
              if(mapgen[x][y] != 0) {

              if(mapgen[x][y] == 0 && charmap[x][y] == 'O') {
                mapgen[x][y] = 1;
              } 
            }


      }
    }
  }


int main() {
  

