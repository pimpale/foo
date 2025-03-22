#%%
import pathlib
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


# load tinystories validation for now (Smaller)
tinystories_texts = pathlib.Path('./data/TinyStories-valid.txt')

#%%



doc = nlp('This is a test and experiment.')
assert not isinstance(doc, list)
for sentence in doc.sentences:
    print(sentence.constituency)
