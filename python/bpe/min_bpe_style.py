#%%

# Note: This algorithm does not handle pretokenization tasks like splitting at spaces

import pathlib

corpus = pathlib.Path('corpus.txt').read_text()

#%%

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    stats = {}
    for a, b in zip(ids, ids[1:]):
        if (a, b) in stats:
            stats[(a, b)] += 1
        else:
            stats[(a, b)] = 1
    return stats


def merge(ids: list[int], mergepair: tuple[int, int], result: int) -> list[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == mergepair:
            new_ids.append(result)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    unencoded_ids = []
    for id in ids:
        unencoded_ids.extend(vocab[id])
    return ''.join(chr(id) for id in unencoded_ids)

def encode(ids: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    # encode earlier ids first
    while len(ids) > 1:
        stats = get_stats(ids)
        # get earliest pair
        pair = min(stats, key=merges.get)
        ids = merge(ids, pair, merges[pair])
            
    return ids

ids = list(corpus.encode('utf-8'))

merges = {}

# dict mapping an id to the actual list of ids
vocab = { id: [id] for id in set(ids) }

n_merges = 100
while len(merges) < n_merges:
    stats = get_stats(ids)
    # get most populous pair:
    mergepair = max(stats, key=stats.get)
    if mergepair is None:
        break
    # coin a new id
    new_id = max(vocab) + 1
    # insert this id into the vocab
    merges[mergepair] = new_id
    vocab[new_id] = vocab[mergepair[0]] + vocab[mergepair[1]]
    # merge the pair
    ids = merge(ids, mergepair, new_id)
    
    print(f'Merged {mergepair} (`{decode([mergepair[0]], vocab)}`, `{decode([mergepair[1]], vocab)}`) into {new_id} (`{decode([new_id], vocab)}`)')
