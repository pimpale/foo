typedef struct TrieNode_s {
   struct TrieNode_s* children[26];
   int count;
} TrieNode;


// returns deepest shared node
int insTrieNode(TrieNode* root, char* str, int depth) {
    // increment count
    root->count++;
    
    // mark ends as terminal
    if(str[0] == 0) {
        return depth;
    }
    
    // a pointer to the pointer to the next node
    TrieNode** nextp = &root->children[str[0]-'a'];
    // fix next pointer if doesnt exist
    if(*nextp == NULL) {
        *nextp = calloc(sizeof(TrieNode), 1);
    }
    
    // go to next
    return insTrieNode(
        *nextp, 
        str+1, 
        root->count > 1 
                       ? depth+1 
                       : depth
    );
}

// free mem
void deleteTrie(TrieNode* root) {
    if(root != NULL) {
        for(int i = 0; i < 26; i++) {
            deleteTrie(root->children[i]);
        }
        free(root);
    }
}

// str must be in root
// buf must be at least as long as the longest shared prefix in the str
void findLongestInTrie(TrieNode* root, char* str, char* buf) {
    int i = 0;
    while(root != NULL && root->count > 1) {
        buf[i] = str[i];
        root = root->children[str[i]-'a'];
    }
}

char * longestCommonPrefix(char ** strs, int strsSize){
  TrieNode* root = calloc(sizeof(TrieNode), 1);
    
  int longestSharedDepth = 0;
  char* longestSharedDepthStr = NULL;
  for(int i = 0; i < strsSize; i++){
      char* str = strs[i];
      int sharedDepth = insTrieNode(root, str, 1);
      // update max
      if(sharedDepth > longestSharedDepth) {
          longestSharedDepth = sharedDepth;
          printf("%i\n", sharedDepth);
          longestSharedDepthStr = str;
      }
  }
  
  char buf[201];
  findLongestInTrie(root, longestSharedDepthStr, buf);
  deleteTrie(root);
  return buf;
}
