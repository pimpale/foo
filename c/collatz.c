#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#define SEARCHMAX 1000000
#define COMPTABLEMAX (SEARCHMAX*1)

uint64_t comptable[COMPTABLEMAX];
uint8_t comptable_visited[COMPTABLEMAX];

uint64_t computeCollatz(uint64_t n)
{
		uint64_t o = n;
		uint64_t c = 0;
		while(n != 1)
		{
				if(n < COMPTABLEMAX && comptable_visited[n])
				{
						c = comptable[n] + c;
						goto exit;
				}
				else
				{
						if(n%2 == 0)
						{
								n = n/2;
						}
						else
						{
								n = 3*n + 1;
						}
						c++;
				}
		}
		
		exit:
		comptable[o] = c;
		comptable_visited[o] = 1;
		return c;
}

int main()
{
		uint64_t max_collatz_steps = 0;
		uint64_t max_collatz_source = 0;
		for(uint64_t n = 1; n < SEARCHMAX; n++)
		{
				int temp = computeCollatz(n);
				if(temp > max_collatz_steps)
				{
						max_collatz_steps = temp;
						max_collatz_source = n;
				}
		}
		printf("%" PRIu64 "\n",max_collatz_source);
		return 0;
}
