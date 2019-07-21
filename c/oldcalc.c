
#include <stdio.h>
#include <stdlib.h>

int stack[500];
size_t stackpos = 0;

#define MAX_TOKEN 5

struct token {
		char token[MAX_TOKEN];
};

int toInt(struct token t)
{
		return atoi(t.token);
}

int isOp(struct token t)
{
		char c = t.token[0];
		return c == '+' || c == '-' || c == '*' || c == '/';
}

int isWhiteSpace(char c)
{
		return c == ' ' || c == '\t' || c == '\n';
}

struct token getNextToken()
{
		int index = 0;
		struct token t;
		unsigned char c = ' ';
		//read till non whitespace char found
		do
		{
				c = (unsigned char) getchar();
		} while(isWhiteSpace(c));

		//read into token until whitespace found
		do 
		{
				t.token[index] = c;
				index++;
				c = (unsigned char) getchar();
		} while(index < MAX_TOKEN && !isWhiteSpace(c));

		return t;
}


int doOp(int n1, int n2, char op)
{
		int res = -1;

		if(op ==  '+')
		{
				res = n1 + n2;
		}
		else if(op ==  '-')
		{
				res = n1 - n2;
		}
		else if(op ==  '*' )
		{
				res = n1 * n2;
		}
		else if(op ==  '/' )
		{
				res = n1 / n2;
		}
		return res;
}


int main() {
		while(1)
		{
				struct token t = getNextToken();

				if(isOp(t))
				{
						char n1 = stack[stackpos];
						stackpos = stackpos - 1;
						char n2 = stack[stackpos];
						stackpos = stackpos - 1;
						int res = doOp(n1,n2,t.token[0]);
						printf("%d\n", res);
				}
				else
				{
						stack[stackpos] = toInt(t);
						stackpos = stackpos + 1;
				}
		}
}

