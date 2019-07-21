#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STACK_MAX 10000
#define NUMERICAL_LITERAL_MAX 3
#define FUNCTION_NAME_MAX 32

#define FATAL(x)        \
  do {                  \
    fprintf(stderr, x); \
    exit(EXIT_FAILURE); \
  } while (0)

int32_t stackpos;
uint8_t stack[STACK_MAX];
void push(uint8_t data) {
  if (stackpos + 1 >= STACK_MAX) {
    FATAL("stack overflow\n");
  }
  stack[stackpos++] = data;
}

uint8_t pop() {
  if (stackpos - 1 < 0) {
    FATAL("stack underflow\n");
  }
  return (stack[stackpos--]);
}

// parse until we hit a closing parentheses
// String format is defined here:
// one byte in each word
// followed by a string length
void parseString(FILE* stream) {
  if (getc(stream) != '(') {
    FATAL("malformed string literal");
  }
  // signal beginning
  push(0);
  int32_t c;
  uint32_t depth = 1;
  while ((c = getc(stream)) != EOF) {
    if (c == '\\') {
      c = getc(stream);
      if (c == EOF) {
        break;
      }
    } else if (c == '(') {
      depth++;
      push((uint8_t)c);
    } else if (c == ')') {
      depth--;
      if (depth == 0) {
        break;
      } else {
        push((uint8_t)c);
      }
    } else {
      push((uint8_t)c);
    }
  }
  // signal end
  push(0);
}

// parse until space encountered, then push number.
// Numbers > 255 result in undefined behavior
void parseNumber(FILE* stream) {
  int32_t c;
  size_t numBufPos = 0;
  char numBuf[NUMERICAL_LITERAL_MAX + 1];
  while ((c = getc(stream)) != EOF) {
    if (!isxdigit(c) || numBufPos >= NUMERICAL_LITERAL_MAX) {
      break;
    } else {
      numBuf[numBufPos++] = (char)c;
    }
  }
  numBuf[NUMERICAL_LITERAL_MAX] = '\0';
  int num = atoi(numBuf);
  if (num > UINT8_MAX || num < 0) {
    FATAL("numerical literal out of bounds");
  }
  push((uint8_t)num);
}

void parse(FILE* stream);

void parseFunction(FILE* stream) {
  char functionBuf[FUNCTION_NAME_MAX + 1];
  size_t len = 0;
  int32_t c;
  while ((c = getc(stream)) != EOF && len <= FUNCTION_NAME_MAX) {
    if (isblank(c) || c == '\n') {
      break;
    } else {
      functionBuf[len++] = (char)c;
    }
  }
  functionBuf[len] = '\0';
  FILE* fp = fopen(functionBuf, "r");
  parse(fp);
  fclose(fp);
}

// Parses stream until end
void parse(FILE* stream) {
  int32_t c;
  while ((c = getc(stream)) != EOF) {
    if (!isblank(c) && c != '\n') {
      ungetc(c, stream);
      if (c == '(' || c == ')') {  // String literal
        parseString(stream);
      } else if (isdigit(c)) {  // Numerical literal
        parseNumber(stream);
      } else {
        parseFunction(stream);
      }
    }
  }
}

int main() {
  FILE* in = fopen("/dev/stdin", "r");
  parse(in);
  fclose(in);
}
