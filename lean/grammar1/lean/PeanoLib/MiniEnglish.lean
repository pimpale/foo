import Lean

open Lean Elab Meta

inductive Noun
| dog | cat | mouse | elephant | person | thing
deriving Repr, BEq, Inhabited

inductive Determiner
| a | an | the
deriving Repr, BEq, Inhabited

inductive Verb
| is | runs | jumps | sleeps | eats
deriving Repr, BEq, Inhabited

inductive Adjective
| big | small | happy | sad | hungry
deriving Repr, BEq, Inhabited

inductive NounPhrase
| simple (det: Determiner) (noun: Noun)
| withAdj (det: Determiner) (adj: Adjective) (noun: Noun)
deriving Repr, BEq, Inhabited

inductive VerbPhrase
| intransitive (verb: Verb)
| withNoun (verb: Verb) (np: NounPhrase)
| withAdj (verb: Verb) (adj: Adjective)
deriving Repr, BEq, Inhabited

inductive Sentence
| statement : NounPhrase → VerbPhrase → Sentence
deriving Repr, BEq, Inhabited

-- Define syntax categories for our mini English grammar
declare_syntax_cat eng_noun
syntax "dog"      : eng_noun
syntax "cat"      : eng_noun
syntax "mouse"    : eng_noun
syntax "elephant" : eng_noun
syntax "person"   : eng_noun
syntax "thing"    : eng_noun

declare_syntax_cat eng_det
syntax "a"        : eng_det
syntax "an"       : eng_det
syntax "the"      : eng_det

declare_syntax_cat eng_verb
syntax "is"       : eng_verb
syntax "runs"     : eng_verb
syntax "jumps"    : eng_verb
syntax "sleeps"   : eng_verb
syntax "eats"     : eng_verb

declare_syntax_cat eng_adj
syntax "big"      : eng_adj
syntax "small"    : eng_adj
syntax "happy"    : eng_adj
syntax "sad"      : eng_adj
syntax "hungry"   : eng_adj

-- Elaboration functions for basic components
def elabNoun : Syntax → MetaM Expr
  | `(eng_noun| dog)      => return .const ``Noun.dog []
  | `(eng_noun| cat)      => return .const ``Noun.cat []
  | `(eng_noun| mouse)    => return .const ``Noun.mouse []
  | `(eng_noun| elephant) => return .const ``Noun.elephant []
  | `(eng_noun| person)   => return .const ``Noun.person []
  | `(eng_noun| thing)    => return .const ``Noun.thing []
  | _ => throwUnsupportedSyntax

def elabDeterminer : Syntax → MetaM Expr
  | `(eng_det| a)   => return .const ``Determiner.a []
  | `(eng_det| an)  => return .const ``Determiner.an []
  | `(eng_det| the) => return .const ``Determiner.the []
  | _ => throwUnsupportedSyntax

def elabVerb : Syntax → MetaM Expr
  | `(eng_verb| is)     => return .const ``Verb.is []
  | `(eng_verb| runs)   => return .const ``Verb.runs []
  | `(eng_verb| jumps)  => return .const ``Verb.jumps []
  | `(eng_verb| sleeps) => return .const ``Verb.sleeps []
  | `(eng_verb| eats)   => return .const ``Verb.eats []
  | _ => throwUnsupportedSyntax

def elabAdjective : Syntax → MetaM Expr
  | `(eng_adj| big)    => return .const ``Adjective.big []
  | `(eng_adj| small)  => return .const ``Adjective.small []
  | `(eng_adj| happy)  => return .const ``Adjective.happy []
  | `(eng_adj| sad)    => return .const ``Adjective.sad []
  | `(eng_adj| hungry) => return .const ``Adjective.hungry []
  | _ => throwUnsupportedSyntax

-- Define syntax categories for phrases
declare_syntax_cat eng_np
syntax eng_det eng_noun : eng_np
syntax eng_det eng_adj eng_noun : eng_np

declare_syntax_cat eng_vp
syntax eng_verb : eng_vp
syntax eng_verb eng_np : eng_vp
syntax eng_verb eng_adj : eng_vp

-- Elaboration for noun phrases
def elabNounPhrase : Syntax → MetaM Expr
  | `(eng_np| $det:eng_det $noun:eng_noun) => do
    let det ← elabDeterminer det
    let noun ← elabNoun noun
    mkAppM ``NounPhrase.simple #[det, noun]
  | `(eng_np| $det:eng_det $adj:eng_adj $noun:eng_noun) => do
    let det ← elabDeterminer det
    let adj ← elabAdjective adj
    let noun ← elabNoun noun
    mkAppM ``NounPhrase.withAdj #[det, adj, noun]
  | _ => throwUnsupportedSyntax

-- Elaboration for verb phrases
def elabVerbPhrase : Syntax → MetaM Expr
  | `(eng_vp| $verb:eng_verb) => do
    let verb ← elabVerb verb
    mkAppM ``VerbPhrase.intransitive #[verb]
  | `(eng_vp| $verb:eng_verb $np:eng_np) => do
    let verb ← elabVerb verb
    let np ← elabNounPhrase np
    mkAppM ``VerbPhrase.withNoun #[verb, np]
  | `(eng_vp| $verb:eng_verb $adj:eng_adj) => do
    let verb ← elabVerb verb
    let adj ← elabAdjective adj
    mkAppM ``VerbPhrase.withAdj #[verb, adj]
  | _ => throwUnsupportedSyntax

-- Define syntax for sentences
declare_syntax_cat eng_sentence
syntax eng_np eng_vp : eng_sentence

-- Elaboration for sentences
def elabSentence : Syntax → MetaM Expr
  | `(eng_sentence| $np:eng_np $vp:eng_vp) => do
    let np ← elabNounPhrase np
    let vp ← elabVerbPhrase vp
    mkAppM ``Sentence.statement #[np, vp]
  | _ => throwUnsupportedSyntax

-- Test elaborators
elab "test_noun " n:eng_noun : term => elabNoun n
elab "test_det " d:eng_det : term => elabDeterminer d
elab "test_verb " v:eng_verb : term => elabVerb v
elab "test_adj " adj:eng_adj : term => elabAdjective adj
elab "test_np " np:eng_np : term => elabNounPhrase np
elab "test_vp " vp:eng_vp : term => elabVerbPhrase vp

-- Main sentence syntax
elab "#eng" s:eng_sentence : term => elabSentence s

-- Examples
#eval test_noun dog
#eval test_det the
#eval test_np the dog
#eval test_np a happy dog
#eval test_vp runs
#eval test_vp eats the mouse

-- Test complete sentences
#eval #eng the dog runs
#eval #eng a happy cat jumps
#eval #eng the hungry elephant eats a small mouse
#eval #eng an elephant is big
