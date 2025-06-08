#!/usr/bin/env python3
"""
Count unique words across all JSON files in a folder.
Usage: python count_words.py <folder_path>
"""

import json
import os
import sys
from pathlib import Path


def count_unique_words(folder_path):
    """Count unique words from all JSON files in the specified folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return None
    
    unique_words = set()
    files_processed = 0
    
    # Find all JSON files in the folder
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'.")
        return 0
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract words from the "words" section if it exists
            if 'words' in data and isinstance(data['words'], dict):
                words = data['words'].keys()
                unique_words.update(words)
                files_processed += 1
                print(f"  - {json_file.name}: {len(data['words'])} words")
            else:
                print(f"  - {json_file.name}: No 'words' section found")
                
        except json.JSONDecodeError as e:
            print(f"  - {json_file.name}: JSON decode error - {e}")
        except Exception as e:
            print(f"  - {json_file.name}: Error - {e}")
    
    print(f"\nProcessed {files_processed} files successfully.")
    return len(unique_words)


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_words.py <folder_path>")
        print("Example: python count_words.py nouns/")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    unique_count = count_unique_words(folder_path)
    
    if unique_count is not None:
        print(f"\nTotal unique words: {unique_count}")


if __name__ == "__main__":
    main() 