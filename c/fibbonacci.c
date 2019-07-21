#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

#define LENGTH 40

int  main() {
	uint64_t arr[LENGTH];

	arr[0] = 0; 
	arr[1] = 1;
	
	//done
	for(int i = 2; i < LENGTH; i++)
	{
		arr[i] = arr[i-1] + arr[i-2];
	}

	//now print
	for(int i = 0; i < LENGTH; i++)
	{
		printf("%" PRIu64 "\n", arr[i]);
	}
}


	
