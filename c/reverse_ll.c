#include <stdio.h>
#include <stdlib.h>

struct Node_s;
typedef struct Node_s {
    int val;
    struct Node_s* next;
} Node;

Node* reverse_ll(Node* head) {
    Node* previous = NULL;
    Node* current = head;
    Node* following = head;

    while(current != NULL) {
        following = following->next;
        current->next = previous;
        previous = current;
        current = following;
    }
    return previous;
}


void print_ll(Node* head) {
    while(head != NULL) {
        printf("%d\n", head->val);
        head = head->next;
    }
}

Node* mknd(int val, Node* next) {
    Node* n = malloc(sizeof(Node));
    n->val = val;
    n->next = next;
    return n;
}

int main(int argc, char** argv) {
  Node* head = mknd(1, mknd(2, mknd(3, mknd(4, NULL))));
  puts("BEFORE REVERSE");
  print_ll(head);
  head = reverse_ll(head);
  puts("AFTER REVERSE");
  print_ll(head);
  return 0;
}
