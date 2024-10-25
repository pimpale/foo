#%%
import pathlib

corpus = pathlib.Path('corpus.txt').read_text()

#%%

def get_stats(ids: list[int], stats: dict[tuple[int, int], int]):
    for a, b in zip(ids, ids[1:]):
        if (a, b) in stats:
            stats[(a, b)] += 1
        else:
            stats[(a, b)] = 1


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

def encode_chunk(ids: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    # encode earlier ids first
    while len(ids) > 1:
        stats = {}
        get_stats(ids, stats)
        # get earliest pair
        pair = min(stats, key=merges.get)
        ids = merge(ids, pair, merges[pair])
    return ids

def encode(corpus: str, merges: dict[tuple[int, int], int]) -> list[int]:
    chunks = pretokenize(corpus)
    idchunks = [list(chunk.encode('utf-8')) for chunk in chunks]
    return [encode_chunk(idchunk, merges) for idchunk in idchunks]

def pretokenize(corpus: str) -> list[str]:
    # first, split the corpus into chunks
    chunks = corpus.split(' ')
    # pad all subsequent chunks with a leading space
    for i in range(1, len(chunks)):
        chunks[i] = ' ' + chunks[i]
    return chunks

if True:
    chunks = pretokenize(corpus)

    idchunks = [list(chunk.encode('utf-8')) for chunk in chunks]

    merges = {}

    # dict mapping an id to the actual list of ids
    idset = set()
    for idchunk in idchunks:
        idset.update(idchunk)
        
    vocab = { id: [id] for id in idset }

    n_merges = 100
    while len(merges) < n_merges:
        stats = {}
        for idchunk in idchunks:
            get_stats(idchunk, stats)
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
        idchunks = [merge(idchunk, mergepair, new_id) for idchunk in idchunks]
        
        print(f'Merged {mergepair} (`{decode([mergepair[0]], vocab)}`, `{decode([mergepair[1]], vocab)}`) into {new_id} (`{decode([new_id], vocab)}`)')
