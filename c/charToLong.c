#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main(char argc, char** argv)
{
	uint8_t charray[] = {0x01, 0xCB, 0x23, 0x11};

	uint64_t thing;
	memcpy(&thing, charray, 4);
	printf("%ld\n", (unsigned long) thing);

	memmove(&thing, charray, 4);
	printf("%ld\n", (unsigned long) thing);

	thing = charray[0] + (charray[1] << 8) + (charray[2] << 16) + (charray[3] << 24);
	printf("%ld\n", (unsigned long) thing);


	thing = (charray[0] << 24) + (charray[1] << 16) + (charray[2] << 8) + charray[3];
	printf("%ld\n", (unsigned long) thing);

	for(int i = 0; i < sizeof(thing); i++)
	{

	}

}
