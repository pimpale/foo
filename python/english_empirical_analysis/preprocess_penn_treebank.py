#%%
import glob
import os
from treebank_parser import Tree, parse_trees, save_trees
from pathlib import Path

# Function to parse an MRG file and return parsed trees
def parse_mrg_file(file_path):
    """Parse an MRG file and return a list of parse trees.
    Each file may contain multiple parse trees (S-expressions)."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    parsed_trees = []
    try:
        all_trees = parse_trees(content)        
        parsed_trees.extend(all_trees)        
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return parsed_trees

# Create a directory for storing processed data
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

# Find all MRG files in the treebank
mrg_files = glob.glob('./data/treebank3/parsed/mrg/wsj/**/*.mrg', recursive=True)

#TODO: temporary, this is for debugging
# mrg_files = glob.glob('./data/treebank3/parsed/mrg/wsj/00/wsj_0001.mrg', recursive=True)

print(f"Found {len(mrg_files)} MRG files.")

# Process each file and collect all trees into a single list
all_parsed_trees = []
file_count = 0
tree_count = 0

for file_path in mrg_files:
    # Parse the file
    parsed_trees = parse_mrg_file(file_path)
    
    # Add all trees from this file to our master list
    all_parsed_trees.extend(parsed_trees)
    
    tree_count += len(parsed_trees)
    file_count += 1
    
    if file_count % 20 == 0:
        print(f"Processed {file_count} files, {tree_count} trees so far")

# Save all trees to a single pickle file
output_file = os.path.join(output_dir, 'all_treebank_trees.pickle')
save_trees(all_parsed_trees, output_file)

print(f"Parsing complete. Processed {file_count} files with a total of {tree_count} parse trees.")
print(f"Results stored in {output_file}")

# Example of how to analyze the data (uncomment to use)
"""
# Basic analysis of the parsed trees
def analyze_trees():
    phrase_counts = {}
    
    for json_file in glob.glob(f"{output_dir}/*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for tree in data['trees']:
            # Count phrase types (labels) at the top level
            label = tree['label']
            phrase_counts[label] = phrase_counts.get(label, 0) + 1
    
    print("Most common phrase types:")
    for phrase, count in sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{phrase}: {count}")

# analyze_trees()
"""


