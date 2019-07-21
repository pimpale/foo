#include <stdio.h>

int main()
{
		int linenum = 0;

		scanf("%d", &linenum);

		for(int i = 0; i < linenum; i++)
		{
				int charnum;
				char c;
				scanf("%d", &charnum);
				scanf("%c", &c);

				for(int a = 0; a < charnum; a++)
				{
						printf("%c", c);
				}
				printf("\n");
		}

		return 0;
}
