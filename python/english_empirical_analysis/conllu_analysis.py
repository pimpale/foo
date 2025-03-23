#%%
import conllu
import pathlib

# Parse the content
sentences = conllu.parse(pathlib.Path('./data/en_ewt-ud-train.conllu').read_text())

# Display some information about the parsed data
print(f"Number of parsed sentences: {len(sentences)}")

print("\nExample sentence:")
print(sentences[0])



def print_dependency_tree(sentence):
    """Print a dependency tree for a CoNLL-U sentence."""
    # Create a mapping of token IDs to their positions
    token_dict = {token['id']: token for token in sentence}
    
    # Find the root(s) of the tree
    roots = [token for token in sentence if token['head'] == 0]
    
    def print_subtree(token_id, indent=0):
        """Recursively print a subtree starting from token_id."""
        if token_id not in token_dict:
            return
            
        token = token_dict[token_id]
        
        # Print current token with its POS tag and dependency relation
        print("  " * indent + f"{token['form']} ({token['upos']}, {token['deprel']})")
        
        # Find and print all children
        children = [t for t in sentence if t['head'] == token_id]
        for child in sorted(children, key=lambda x: x['id']):
            print_subtree(child['id'], indent + 1)
    
    # Print each root and its subtree
    for root in roots:
        print_subtree(root['id'])


# Print syntax trees for a few example sentences
print("\nSyntax Trees:")
for i, sentence in enumerate(sentences[:3]):  # Limiting to first 3 sentences
    print(f"\nSentence {i+1}:")
    print(f"Text: {' '.join(token['form'] for token in sentence)}")
    print("\nDependency Tree:")
    print_dependency_tree(sentence)

print("\n(Showing only the first 3 sentences for brevity)")

