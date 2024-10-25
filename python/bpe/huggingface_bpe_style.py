#%%
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from collections import defaultdict


def pretokenize(text: str) -> list[str]:
    words = text.split()
    return [f"#{word}" for word in words]

word_freq = defaultdict(int)
for text in corpus:
    for word in pretokenize(text):
        word_freq[word] += 1
        
alphabet = []

for word in word_freq:
    for char in word:
        if char not in alphabet:
            alphabet.append(char)

alphabet.sort()

vocab = ["<|endoftext|>"] + alphabet.copy()

def compute_pair_freqs(word_freqs, splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        chars = splits[word]
        for pair in zip(chars, chars[1:]):
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, word_freqs, splits):
    new_splits = {}
    for word in word_freqs.keys():
        split = splits[word]
        if len(split) < 2:
            new_splits[word] = split
            continue
        
        new_split = []
        i = 0
        while i < len(split):
            if i + 1 < len(split) and split[i] == a and split[i + 1] == b:
                new_split.append(a + b)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split
    return new_splits

def find_best_pair(pair_freqs):
    return max(pair_freqs, key=pair_freqs.get)

splits = {
    word: [char for char in word]
    for word in word_freq.keys()
}
merges = {}
while len(vocab) < 50:
    pair_freqs = compute_pair_freqs(word_freq, splits)
    best_pair = find_best_pair(pair_freqs)
    if pair_freqs[best_pair] < 2:
        break
    a, b = best_pair
    print(f"Merge pair: {a} + {b}")
    merges[best_pair] = a + b
    vocab.append(a + b)
    splits = merge_pair(a, b, word_freq, splits)
    
    
def tokenize(text: str, merges: dict[tuple[str, str], str]) -> list[str]:
    pre_tokenize_text = pretokenize(text)
    splits = [[l for l in word] for word in pre_tokenize_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split
    splitsum = []
    for split in splits:
        splitsum += split
    return splitsum