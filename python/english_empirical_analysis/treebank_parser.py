from dataclasses import dataclass
from lark import Lark, Transformer
import pickle

@dataclass
class Tree:
    label: str
    children: list["Tree"]
    
    def __repr__(self) -> str:
        children_repr = " ".join(repr(c) for c in self.children)
        if len(children_repr) > 0:
            return f"({self.label} {children_repr})"
        else:
            return self.label

def parse_trees(s) -> list[Tree]:
    """
    Parse a string containing one or more Penn Treebank trees.
    Returns a single Tree if there's only one tree, or a list of Trees if there are multiple.
    """
    # Define the grammar for Penn Treebank S-expressions
    grammar = r"""
        start: treebank_tree+
        
        // Handles the outer parentheses wrapping each tree
        treebank_tree: "(" tree ")"
        
        tree: "(" LABEL [tree+] ")" -> node
            | TERMINAL -> leaf
            
        LABEL: /[^\s()]+/
        TERMINAL: /[^\s()]+/
        
        %import common.WS
        %ignore WS
    """
    
    # Define a transformer to convert Lark's tree to our Tree structure
    class TreeTransformer(Transformer):
        def start(self, items):
            # Return the list of trees directly
            return items
        
        def treebank_tree(self, items):
            # Extract the actual tree from inside the outer parentheses
            return items[0]
            
        def node(self, items):
            label = items[0].value
            children = items[1:] if len(items) > 1 else []
            return Tree(label, children)
            
        def leaf(self, items):
            return Tree(items[0].value, [])
            
    # Create the parser and parse the input
    parser = Lark(grammar, parser='lalr', transformer=TreeTransformer())
    trees = parser.parse(s)    
    return trees

def save_trees(trees, filename):
    """
    Save a tree or list of trees to a pickle file.
    
    Args:
        trees: A single Tree object or a list of Tree objects
        filename: Path where the pickle file will be saved
    """
    with open(filename, 'wb') as f:
        pickle.dump(trees, f)

def load_trees(filename):
    """
    Load a tree or list of trees from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        A single Tree object or a list of Tree objects, depending on what was saved
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
