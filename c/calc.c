#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_STACK 500

void* stack[MAX_STACK];
size_t stackpos = 0;

void push(void* data, size_t len)
{
  if(stackpos + len >= MAX_STACK) {
    printf("stack overflow\n");
    exit(EXIT_FAILURE);
  }
  memmove(&(stack[stackpos]), data, len);
  stackpos += len;
}

void pop(void* data, size_t len)
{
  if(stackpos - len < 0)
  {
    printf("stack underflow\n");
    exit(EXIT_FAILURE);
  }
  stackpos-=len;
  memmove(data, &(stack[stackpos]), len);
}

bool tokenDataEqual(Token t, uint8_t* str)
{
  return (strncmp(t.data, str, MAX_TOKEN) == 0);
}

bool isWhiteSpace(uint8_t c) { return c == ' ' || c == '\t' || c == '\n'; }


// TODO by default, there are 3 modes. String mode, number mode, and command mode.
// Number mode is trigged by a number at the beginning of a token and ends at whitespace
// A number primitive is stored in numerical string format (whether double, float, or int)
// String mode is triggered by a " a the beginning of a line. It will continue until another " is encountered
// it will be stored (one char in each box) in the token without the quotes
// Command mode is triggered by a : at the beginning of a token, continuing till a whitespace
// It will be stored without the : in the data of a token.
Token getNextToken(FILE* fp) {
  int index = 0;
  Token t = {0};
  char c = ' ';
  // read till non whitespace uint8_t found
  do {
    c = (char)fgetc(fp);
  } while (isWhiteSpace(c));

  /* Number literal */
  if(isdigit(c))
  {
    int index = 0;
    char numBuf[MAX_TOKEN];
    do {
      numBuf[index] = c;
      index++;
      c = getchar();
    } while(index < MAX_TOKEN && !isWhiteSpace(c));
    int res = atoi(numBuf);
    t.len = sizeof(int);
    memcpy(t.data, &res, t.len); 
    t.type = TOKEN_TYPE_NUMBER_LITERAL;
  } else if (c == '(') {
    /* The depth of the parenthetical statement */
    int depth = 0;
    /* String literal */
    int index = 0;
    char strBuf[MAX_TOKEN];
    while (true) {
      if(index +1 >= MAX_TOKEN)
      {
        printf("malformed string literal: too long\n");
        exit(EXIT_FAILURE);
      }
      c = (char) getchar();
      if(c == '"')
      {
        break;
      }
      strBuf[index] = c;
      index++;
    }
    t.len = index + 1;
    memcpy(t.data, strBuf, t.len);
    t.data[index] = '\0';
    t.type = TOKEN_TYPE_STRING_LITERAL;
  } else if (c == ':') {
    /* Function name */
    int index = 0;
    char strBuf[MAX_TOKEN];
    while (true) {
      if(index +1 >= MAX_TOKEN)
      {
        printf("malformed function name: too long\n");
        exit(EXIT_FAILURE);
      }
      c = (char) getchar();
      if(isWhiteSpace(c))
      {
        break;
      }
      strBuf[index] = c;
      index++;
    }
    t.len = index + 1;
    memcpy(t.data, strBuf, t.len);
    t.data[index] = '\0';
    t.type = TOKEN_TYPE_FUNCTION;
  }
  return t;
}

//Figure out what needs to be done in terms of pushing a string to the stack (
void execToken(Token t) {
  if(tokenDataEqual(t, "."));
    Token printable;
    popToken(&printable);
    if(printable.type == TOKEN_TYPE_NUMBER_LITERAL) {
      printf("%d", *((int*)printable.data));
    } else if(printable.type == TOKEN_TYPE_STRING_LITERAL) {
      puts(printable.data);
    }
  } else if(
  if(tokenEqual(t, "+")) {
    uint8_t a = pop();
    uint8_t b = pop();
    push(a+b);
  } else if(tokenEqual(t, "-")) {
    uint8_t a = pop();
    uint8_t b = pop();
    push(a-b);
  } else if(tokenEqual(t, "dup")) {
    uint8_t a = pop();
    push(a);
    push(a);
  } else if(tokenEqual(t, "drop")) {
    pop();
  } else if(tokenEqual(t, ":")) {
    
  } else if(tokenEqual(t, ".")) {
    printf("%d", pop());
  }
}

int main() {
  while (1) {
    Token t = getNextToken();
    if(t.type == TOKEN_TYPE_FUNCTION)
    {
      execToken(t);
    }
    else
    {
      if(
    }
  }

}

