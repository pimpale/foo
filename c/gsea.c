#include <stdio.h>
#include <stdlib.h>

#define GENE_COUNT 5000

int main() {
  int* RankedGeneSet;
  RankedGeneSet = malloc(GENE_COUNT);

  int runningSum = 0;
  for(size_t i = 0; i < GENE_COUNT; i++) {
    int gene = RankedGeneSet[i];
    if(isInGeneSet(gene)) {
      runningSum+=50;
    } else {
      runningSum--;
    }

    return 0;
}


    
