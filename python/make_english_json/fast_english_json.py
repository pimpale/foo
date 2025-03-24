#%%
import json
import pathlib
from copy import deepcopy

indeclinable = json.loads(pathlib.Path('indeclinable.json').read_text())
nouns = json.loads(pathlib.Path('nouns.json').read_text())
irregular_nouns = json.loads(pathlib.Path('irregular_nouns.json').read_text())
verbs = json.loads(pathlib.Path('verbs.json').read_text())
irregular_verbs = json.loads(pathlib.Path('irregular_verbs.json').read_text())
prepositions = json.loads(pathlib.Path('prepositions.json').read_text())


def noun_to_noun_pl(singular: str) -> str:
    if singular in irregular_nouns:
        return irregular_nouns[singular]['plural']
    
    if singular.endswith('s') or singular.endswith('sh') or singular.endswith('ch') or singular.endswith('x'):
        return singular + 'es'
    elif singular.endswith('y'):
        # if the letter before 'y' is a vowel, add 's'
        if singular[-2] in 'aeiou':
            return singular + 's'
        else:
            return singular[:-1] + 'ies'
    else:
        return singular + 's'

def vb_to_vbd(verb: str) -> str:
    """
    Convert a verb to its preterite form.
    ex: 'scratch' -> 'scratched'
    """
    if verb in irregular_verbs:
        return irregular_verbs[verb]['VBD']
    
    if verb.endswith('e'):
        return verb + 'd'
    elif verb.endswith('y'):
        # if the letter before 'y' is a vowel, add 'ed'
        if verb[-2] in 'aeiou':
            return verb + 'ed'
        else:
            return verb[:-1] + 'ied'
    else:
        return verb + 'ed'

def vb_to_vbn(verb: str) -> str:
    """
    Convert a verb to its past participle form.
    ex: 'scratch' -> 'scratched'
    """
    
    if verb in irregular_verbs:
        return irregular_verbs[verb]['VBN']
    
    return vb_to_vbd(verb)

def vb_to_vbg(verb: str) -> str:
    """
    Convert a verb to its gerund form.
    ex: 'run' -> 'running'
    """
    
    if verb in irregular_verbs:
        return irregular_verbs[verb]['VBG']
    
    if verb.endswith('e'):
        return verb[:-1] + 'ing'
    elif verb.endswith('ie'):
        return verb[:-2] + 'ying'
    else:
        return verb + 'ing'
    
    
def vb_to_vbz(verb: str) -> str:
    """
    Convert a verb to its third person singular form.
    ex: 'run' -> 'runs'
    """
    
    if verb in irregular_verbs:
        return irregular_verbs[verb]['VBZ']
    
    if verb.endswith('s') or verb.endswith('sh') or verb.endswith('ch') or verb.endswith('x'):
        return verb + 'es'
    elif verb.endswith('y'):
        # if the letter before 'y' is a vowel, add 's'
        if verb[-2] in 'aeiou':
            return verb + 's'
        else:
            return verb[:-1] + 'ies'
    else:
        return verb + 's'

def vb_to_vbp(verb: str) -> str:
    """
    Convert a verb to its third person plural form.
    ex: 'run' -> 'run'
    """
    
    if verb in irregular_verbs:
        return irregular_verbs[verb]['VBP']
    
    return verb

# Create a new json file with all the forms of the words

english_json = {}

for kind in indeclinable:
    english_json[kind] = deepcopy(indeclinable[kind])
    

# add uncountable nouns
english_json['uncountable_noun'] = deepcopy(nouns['uncountable_noun'])

# add countable nouns
english_json['noun'] = deepcopy(nouns['noun'])
# add countable plural nouns
for noun in nouns['noun']:
    english_json['noun'][noun_to_noun_pl(noun)] = None
    
for kind in verbs:   
    # add vb form
    english_json[kind] = deepcopy(verbs[kind])
   
    # add vbd form
    kind_vbd = kind.replace('vb', 'vbd', 1)
    english_json[kind_vbd] = {vb_to_vbd(verb): None for verb in verbs[kind]}
    # add vbn form
    kind_vbn = kind.replace('vb', 'vbn', 1)
    english_json[kind_vbn] = {vb_to_vbn(verb): None for verb in verbs[kind]}
    # add vbg form
    kind_vbg = kind.replace('vb', 'vbg', 1)
    english_json[kind_vbg] = {vb_to_vbg(verb): None for verb in verbs[kind]}
    # add vbz form
    kind_vbz = kind.replace('vb', 'vbz', 1)
    english_json[kind_vbz] = {vb_to_vbz(verb): None for verb in verbs[kind]}
    # add vbp form
    kind_vbp = kind.replace('vb', 'vbp', 1)
    english_json[kind_vbp] = {vb_to_vbp(verb): None for verb in verbs[kind]}
    
# add prepositions
for kind in prepositions:
    english_json[kind] = deepcopy(prepositions[kind])
# join all classes of preposition together (not including duplicates) and store this too
english_json['preposition'] = {}
for kind in prepositions:
    for prep in prepositions[kind]:
        english_json['preposition'][prep] = None
    
pathlib.Path('english.json').write_text(json.dumps(english_json, indent=4))