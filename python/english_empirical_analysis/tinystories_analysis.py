#%%
import pathlib
import stanza
import itertools
from lemminflect import getLemma

# setup pipelien
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package='default_accurate')

# load tinystories validation for now (Smaller)
tinystories_texts = pathlib.Path('./data/TinyStories-valid.txt').read_text().split('<|endoftext|>')

#%%
sentences = []
for tinystory in tinystories_texts[:100]:
    doc = nlp(tinystory)
    assert not isinstance(doc, list)
    sentences += [s.constituency for s in doc.sentences]

# %%

def get_all_of_label(label, constituent):
    vps = []
    if constituent.label == label:
        vps.append(constituent)
    for child in constituent.children:
        vps.extend(get_all_of_label(label, child))
    
    return vps

vps = list(itertools.chain(*[get_all_of_label('VP', sentence) for sentence in sentences]))

# Function to count NPs within a constituent
def count_top_level_nps(constituent): 
    return sum(1 for child in constituent.children if child.label == 'NP')

# Function to extract the main verb from a VP
def extract_verb_lemma(vp):
    # Find the head verb (typically the leftmost V* in a VP)
    for child in vp.children:
        if isinstance(child, str):
            continue
        if child.label.startswith('VB'):
            # Return the text of the verb
            return child.children[0].label.lower()
        # Recursively check inside nested constituents
        if not isinstance(child, str):
            verb = extract_verb_lemma(child)
            if verb:
                return verb
    return None

# Categorize VPs
intransitive_vps = []
transitive_vps = []
ditransitive_vps = []

for vp in vps:
    np_count = count_top_level_nps(vp)
    if np_count == 0:
        intransitive_vps.append(vp)
    elif np_count == 1:
        transitive_vps.append(vp)
    elif np_count == 2:
        ditransitive_vps.append(vp)

# Extract lemmas and store in sets
intransitive_lemmas = set()
transitive_lemmas = set()
ditransitive_lemmas = set()

# Helper function to get lemma using lemminflect
def get_lemma(word):
    lemmas = getLemma(word, upos='VERB')
    return lemmas[0] if lemmas else word

for vp in intransitive_vps:
    verb = extract_verb_lemma(vp)
    if verb:
        intransitive_lemmas.add(get_lemma(verb))

for vp in transitive_vps:
    verb = extract_verb_lemma(vp)
    if verb:
        transitive_lemmas.add(get_lemma(verb))

for vp in ditransitive_vps:
    verb = extract_verb_lemma(vp)
    if verb:
        ditransitive_lemmas.add(get_lemma(verb))

# Print results
print(f"Intransitive verbs ({len(intransitive_vps)} VPs, {len(intransitive_lemmas)} unique lemmas):")
print(intransitive_lemmas)
print("\n")

print(f"Transitive verbs ({len(transitive_vps)} VPs, {len(transitive_lemmas)} unique lemmas):")
print(transitive_lemmas)
print("\n")

print(f"Ditransitive verbs ({len(ditransitive_vps)} VPs, {len(ditransitive_lemmas)} unique lemmas):")
print(ditransitive_lemmas)