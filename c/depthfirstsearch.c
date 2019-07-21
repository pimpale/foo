#include <stdio.h>

#define MAX_TREE_DEPTH 256

typedef struct Node {
	// put something else or whatever here
	int value;
	struct Node* left;
	struct Node* right;
} Node;

typedef struct {
	int depth;
	Node* ppNodeArray[MAX_TREE_DEPTH];
} NodeStack;

void init_NodeStack(NodeStack* pNodeStack) { pNodeStack->depth = 0; }

void push(NodeStack* pNodeStack, Node* pNode) {
	if (pNodeStack->depth >= MAX_TREE_DEPTH) {
		// Stack at capacity
		return;
	}
	// increase the size of the stack
	pNodeStack->depth++;
	// set the node at that point
	pNodeStack->ppNodeArray[pNodeStack->depth] = pNode;
}

Node* pop(NodeStack* pNodeStack) {
	if (pNodeStack->depth <= 0) {
		// stack empty
		return NULL;
	}
	pNodeStack->depth--;
	return pNodeStack->ppNodeArray[pNodeStack->depth];
}

void iter(Node* pRootNode) {
	// initialize stack
	NodeStack nodeStack;
	init_NodeStack(&nodeStack);

	// put our given node upon the stack
	push(&nodeStack, pRootNode);

	while (nodeStack.depth > 0) {
		Node* pCurrentNode = pop(&nodeStack);
		if (pCurrentNode == NULL) {
			continue;
		}
		push(&nodeStack, pCurrentNode->left);
		push(&nodeStack, pCurrentNode->right);
		// DO SOMETHING TO CURRENT NODE HERE
		printf("Value of node is %d\n", pCurrentNode->value);
	}
}

int main() {}

