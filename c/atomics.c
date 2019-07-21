#include <stdbool.h>
#include <stdatomic.h>

static atomic_bool test = false;


int main()
{
	if(!test)
	{
		test = true;
	}
}
