#!/usr/bin/env python3
import sys, os, json

# Ensure the api directory is on the path
sys.path.insert(0, os.path.abspath('api'))

from verbnet import VerbNetParser

# Define helper functions to normalize and validate primary frames
def normalize_primary(primary):
    """Strip annotations, merge that-S, and drop trailing adjuncts from a primary frame."""
    # Strip annotations
    raw = [p.split('.')[0] for p in primary]
    # Merge 'that' + 'S' into a single 'that_S' slot
    slots = []
    i = 0
    while i < len(raw):
        if raw[i] == 'that' and i + 1 < len(raw) and raw[i+1] == 'S':
            slots.append('that_S')
            i += 2
        elif raw[i] in ('NP', 'V', 'ADJ', 'ADV', 'S', 'S-Quote', 'S_INF'):
            slots.append(raw[i])
            i += 1
        else:
            break

    return slots

def is_valid_primary(slots):
    """Return True if slots represent NP-V, NP-V-X, or NP-V-NP-NP (where X is NP, ADJ, ADV, S, or that_S)."""
    # Must start with NP V and have length 2, 3, or 4
    if not (len(slots) in (2, 3, 4) and slots[0] == 'NP' and slots[1] == 'V'):
        return False
    if len(slots) == 2:
        return True  # NP-V
    if len(slots) == 3:
        # Single complement: NP, ADJ, ADV, S, S-Quote, or that_S
        return slots[2] in ('NP', 'ADJ', 'ADV', 'S', 'S-Quote', 'that_S', 'S_INF')
    # Ditransitive: NP V NP NP
    if len(slots) == 4:
        return slots[2] == 'NP' and slots[3] in ('NP', 'ADJ', 'ADV', 'S', 'S-Quote', 'that_S', 'S_INF')
    return False

def main():
    # Initialize the parser for VerbNet 3.4
    vnp = VerbNetParser(version='3.4')
    verb_to_primaries = {}

    # Collect raw primary patterns per verb
    for vc in vnp.get_verb_classes():
        if not vc.members:
            continue
        for frame in vc.frames:
            primary = frame.primary
            for member in vc.members:
                verb = member.name
                primaries = verb_to_primaries.setdefault(verb, [])
                if primary not in primaries:
                    primaries.append(primary)

    # Simplify to NP-V, NP-V-NP, NP-V-NP-NP patterns
    simple = {}
    for verb, primaries in verb_to_primaries.items():
        simple_frames = set()
        for primary in primaries:
            slots = normalize_primary(primary)
            if is_valid_primary(slots):
                simple_frames.add(tuple(slots))
            if verb == 'write':
                print(primary, slots)
        if simple_frames:
            # Sort by length then lexicographically for consistency
            simple[verb] = [list(s) for s in sorted(simple_frames, key=lambda x: (len(x), x))]

    # Print the simplified JSON mapping
    print(json.dumps(simple, indent=2))

if __name__ == '__main__':
    main() 