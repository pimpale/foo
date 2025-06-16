@preprocessor module

@{%
import english from './english.json';

function isPoS(pos) {
  return {test: word => (english[word] ?? ["UNK"]).includes(pos)}
}

function isAnyOfPoS(pos_arr) {
  return {test: word => pos_arr.some(pos => (english[word] ?? ["UNK"]).includes(pos))}
}

// parts of speech
const det = isPoS("det");
const pronoun = isPoS("pronoun");
const possessive_pronoun = isPoS("possessive_pronoun");
const possessive_adjective = isPoS("possessive_adjective");
const proper_noun = isPoS("proper_noun");
const uncountable_noun = isPoS("uncountable_noun");
const noun = isPoS("noun");
const preposition = isPoS("preposition");

// particles
const to = isPoS("to");
const s = isPoS("s");
const not = isPoS("not");
const that = isPoS("that");
const interrogative_subordinator = isPoS("interrogative_subordinator");
const how = isPoS("how");
const why = isPoS("why");
const which = isPoS("which");

// punctuation
const period = {test: x => x == "." };
const question_mark = { test: x => x == "?" };
const exclamation_mark = { test: x => x == "!" };

// verbs

// Modal (MODAL)
const modal = isPoS("modal");

// Base Verb (VB)
const vb = isPoS("vb");
const vb_ap = isPoS("vb_ap");
const vb_to_inf_cl = isPoS("vb_to_inf_cl");
const vb_bare_inf_cl = isPoS("vb_bare_inf_cl");
const vb_declarative_cl = isPoS("vb_declarative_cl");
const vb_exclamative_cl = isPoS("vb_exclamative_cl");
const vb_interrogative_cl = isPoS("vb_interrogative_cl");
const vb_vbg_cl = isPoS("vb_vbg_cl");
const vb_vbn_cl = isPoS("vb_vbn_cl");
const vb_np = isPoS("vb_np");
const vb_np_ap = isPoS("vb_np_ap");
const vb_np_to_inf_cl = isPoS("vb_np_to_inf_cl");
const vb_np_bare_inf_cl = isPoS("vb_np_bare_inf_cl");
const vb_np_declarative_cl = isPoS("vb_np_declarative_cl");
const vb_np_exclamative_cl = isPoS("vb_np_exclamative_cl");
const vb_np_interrogative_cl = isPoS("vb_np_interrogative_cl");
const vb_np_np = isPoS("vb_np_np");

// Verb Gerund or Present Participle (VBG)
const vbg = isPoS("vbg");
const vbg_ap = isPoS("vbg_ap");
const vbg_to_inf_cl = isPoS("vbg_to_inf_cl");
const vbg_bare_inf_cl = isPoS("vbg_bare_inf_cl");
const vbg_declarative_cl = isPoS("vbg_declarative_cl");
const vbg_exclamative_cl = isPoS("vbg_exclamative_cl");
const vbg_interrogative_cl = isPoS("vbg_interrogative_cl");
const vbg_vbg_cl = isPoS("vbg_vbg_cl");
const vbg_vbn_cl = isPoS("vbg_vbn_cl");
const vbg_np = isPoS("vbg_np");
const vbg_np_ap = isPoS("vbg_np_ap");
const vbg_np_to_inf_cl = isPoS("vbg_np_to_inf_cl");
const vbg_np_bare_inf_cl = isPoS("vbg_np_bare_inf_cl");
const vbg_np_declarative_cl = isPoS("vbg_np_declarative_cl");
const vbg_np_exclamative_cl = isPoS("vbg_np_exclamative_cl");
const vbg_np_interrogative_cl = isPoS("vbg_np_interrogative_cl");
const vbg_np_np = isPoS("vbg_np_np");

// Verb Past Participle (VBN)
const vbn = isPoS("vbn");
const vbn_ap = isPoS("vbn_ap");
const vbn_to_inf_cl = isPoS("vbn_to_inf_cl");
const vbn_bare_inf_cl = isPoS("vbn_bare_inf_cl");
const vbn_declarative_cl = isPoS("vbn_declarative_cl");
const vbn_exclamative_cl = isPoS("vbn_exclamative_cl");
const vbn_interrogative_cl = isPoS("vbn_interrogative_cl");
const vbn_vbg_cl = isPoS("vbn_vbg_cl");
const vbn_vbn_cl = isPoS("vbn_vbn_cl");
const vbn_np = isPoS("vbn_np");
const vbn_np_ap = isPoS("vbn_np_ap");
const vbn_np_to_inf_cl = isPoS("vbn_np_to_inf_cl");
const vbn_np_bare_inf_cl = isPoS("vbn_np_bare_inf_cl");
const vbn_np_declarative_cl = isPoS("vbn_np_declarative_cl");
const vbn_np_exclamative_cl = isPoS("vbn_np_exclamative_cl");
const vbn_np_interrogative_cl = isPoS("vbn_np_interrogative_cl");
const vbn_np_np = isPoS("vbn_np_np");

// Verb Finite (VBF): supercategory for the following
const vbf = isAnyOfPoS(["vbd","vbp","vbz"]);
const vbf_ap = isAnyOfPoS(["vbd_ap","vbp_ap","vbz_ap"]);
const vbf_to_inf_cl = isAnyOfPoS(["vbd_to_inf_cl","vbp_to_inf_cl","vbz_to_inf_cl"]);
const vbf_bare_inf_cl = isAnyOfPoS(["vbd_bare_inf_cl","vbp_bare_inf_cl","vbz_bare_inf_cl"]);
const vbf_declarative_cl = isAnyOfPoS(["vbd_declarative_cl","vbp_declarative_cl","vbz_declarative_cl"]);
const vbf_exclamative_cl = isAnyOfPoS(["vbd_exclamative_cl","vbp_exclamative_cl","vbz_exclamative_cl"]);
const vbf_interrogative_cl = isAnyOfPoS(["vbd_interrogative_cl","vbp_interrogative_cl","vbz_interrogative_cl"]);
const vbf_vbg_cl = isAnyOfPoS(["vbd_vbg_cl","vbp_vbg_cl","vbz_vbg_cl"]);
const vbf_vbn_cl = isAnyOfPoS(["vbd_vbn_cl","vbp_vbn_cl","vbz_vbn_cl"]);
const vbf_np = isAnyOfPoS(["vbd_np","vbp_np","vbz_np"]);
const vbf_np_ap = isAnyOfPoS(["vbd_np_ap","vbp_np_ap","vbz_np_ap"]);
const vbf_np_to_inf_cl = isAnyOfPoS(["vbd_np_to_inf_cl","vbp_np_to_inf_cl","vbz_np_to_inf_cl"]);
const vbf_np_bare_inf_cl = isAnyOfPoS(["vbd_np_bare_inf_cl","vbp_np_bare_inf_cl","vbz_np_bare_inf_cl"]);
const vbf_np_declarative_cl = isAnyOfPoS(["vbd_np_declarative_cl","vbp_np_declarative_cl","vbz_np_declarative_cl"]);
const vbf_np_exclamative_cl = isAnyOfPoS(["vbd_np_exclamative_cl","vbp_np_exclamative_cl","vbz_np_exclamative_cl"]);
const vbf_np_interrogative_cl = isAnyOfPoS(["vbd_np_interrogative_cl","vbp_np_interrogative_cl","vbz_np_interrogative_cl"]);
const vbf_np_np = isAnyOfPoS(["vbd_np_np","vbp_np_np","vbz_np_np"]);

// certain core verbs
const be_fin = isAnyOfPoS(["is","are","were"]);
const do_fin = isAnyOfPoS(["do","does","did"]);

// adjectives
const adj = isPoS("adj");
const adj_pp = isPoS("adj_pp");
const adj_declarative_cl = isPoS("adj_declarative_cl");
const adj_to_inf_cl = isPoS("adj_to_inf_cl");

// adverbs
const adv = isPoS("adv");
const precorenp_modifier = isPoS("precorenp_modifier");
const postcorenp_modifier = isPoS("postcorenp_modifier");

// wh-words (that replace nouns)
const wh = isPoS("wh");

// define postprocessors

// nonterminal 
function nt(kind) {
    return (children) => ({kind, children});
}

// nonterminal 
function nonterminal_unpack(kind) {
    return ([children]) => ({kind, children});
}

// terminal
function t(kind) {
    return ([value]) => ({kind, children: value});
}

%}

text -> sentence:* {%nonterminal_unpack("text")%}

sentence -> 
      fin_cl period {%nt("sentence")%}
    | fin_cl exclamation_mark {%nt("sentence")%}
    | question_cl question_mark {%nt("sentence")%}

# a declarative finite clause
fin_cl -> adjunct_list np fin_vp {%nt("fin_cl")%}

# a question clause
question_cl ->
                subj_aux_inv_cl          {%nt("question_cl")%} # are you happy?
    | wh        fin_vp                   {%nt("question_cl")%} # who ate that?
    | wh        subj_aux_inv_cl_np_moved {%nt("question_cl")%} # what did you eat?
    | why       subj_aux_inv_cl          {%nt("question_cl")%}
    | how advp? subj_aux_inv_cl          {%nt("question_cl")%} # how did you eat the apple?
    | how       subj_aux_inv_cl_ap_moved {%nt("question_cl")%} # how happy are you?


subj_aux_inv_cl ->
# modal
      modal         not? np bare_inf_cl      {%nt("subj_aux_inv_cl")%} # can you eat?
# finite
    | be_fin        not? np ap               {%nt("subj_aux_inv_cl")%} # were you happy? (`be` when used as a copula)
    | be_fin        not? np np               {%nt("subj_aux_inv_cl")%} # were you a watchman? (`be` when used as an equative)
    | vbf_vbg_cl    not? np vbg_cl           {%nt("subj_aux_inv_cl")%} # were you eating?
    | vbf_vbn_cl    not? np vbn_cl           {%nt("subj_aux_inv_cl")%} # were you eaten? / had you eaten?
    | do_fin        not? np bare_inf_cl      {%nt("subj_aux_inv_cl")%} # did you eat?


subj_aux_inv_cl_np_moved ->
# modal (move from head)
      modal         not? bare_inf_cl                  {%nt("subj_aux_inv_cl_np_moved")%} # who [can sing]?
# modal (move from argument)
    | modal         not? np bare_inf_cl_np_moved      {%nt("subj_aux_inv_cl_np_moved")%} # what [can you sing]?
# finite (move from head)
    | be_fin        not? ap                           {%nt("subj_aux_inv_cl_np_moved")%} # who [was happy]? (`be` when used as a copula)
    | vbf_vbg_cl    not? vbg_cl                       {%nt("subj_aux_inv_cl_np_moved")%} # who [was eating]?
    | vbf_vbn_cl    not? vbn_cl                       {%nt("subj_aux_inv_cl_np_moved")%} # who [was eaten]? / who [had eaten]?
    | do_fin        not? bare_inf_cl                  {%nt("subj_aux_inv_cl_np_moved")%} # who [didn't eat]?
# finite (move from argument)
    | vbf_vbg_cl    not? np vbg_cl_np_moved           {%nt("subj_aux_inv_cl_np_moved")%} # what [were you eating]?
    | vbf_vbn_cl    not? np vbn_cl_np_moved           {%nt("subj_aux_inv_cl_np_moved")%} # what [were you given]? / what [had you eaten]?
    | do_fin        not? np bare_inf_cl_np_moved      {%nt("subj_aux_inv_cl_np_moved")%} # what [did you eat]?
# finite (move from )

subj_aux_inv_cl_ap_moved ->
      modal         not? np bare_inf_cl_ap_moved      {%nt("subj_aux_inv_cl_ap_moved")%} # how [can you eat]?
# finite
    | be_fin        not? np                           {%nt("subj_aux_inv_cl_ap_moved")%} # how [were you]? (`be` when used as a copula)
    | vbf_vbg_cl    not? np vbg_cl_ap_moved           {%nt("subj_aux_inv_cl_ap_moved")%} # how [were you feeling]?
    | vbf_vbn_cl    not? np vbn_cl_ap_moved           {%nt("subj_aux_inv_cl_ap_moved")%} # how [were you found]? / how [had you felt]?
    | do_fin        not? np bare_inf_cl_ap_moved      {%nt("subj_aux_inv_cl_ap_moved")%} # how [did you feel]?


# following constituents are flat grammars that permit core arguments + some adjuncts in any order
adjunct_list -> adjunct:* {%nonterminal_unpack("adjunct_list")%}

adjunct_list_ap ->
      adjunct     adjunct_list_ap      {%nt("adjunct_list_ap")%}
    | ap          adjunct_list         {%nt("adjunct_list_ap")%}

adjunct_list_to_inf_cl ->
      adjunct     adjunct_list_to_inf_cl  {%nt("adjunct_list_to_inf_cl")%}
    | to_inf_cl   adjunct_list            {%nt("adjunct_list_to_inf_cl")%}

# adjuncts also don't seem to sit before the bare infintive 
# - *i helped before dawn clean the car and the house 
# - *i made you earlier stop talking about this very long phrase 
adjunct_list_bare_inf_cl ->
      bare_inf_cl   adjunct_list         {%nt("adjunct_list_bare_inf_cl")%}

adjunct_list_declarative_cl ->
      adjunct          adjunct_list_declarative_cl      {%nt("adjunct_list_declarative_cl")%}
    | declarative_cl   adjunct_list                     {%nt("adjunct_list_declarative_cl")%}

adjunct_list_exclamative_cl ->
      adjunct     adjunct_list_exclamative_cl           {%nt("adjunct_list_exclamative_cl")%}
    | exclamative_cl   adjunct_list                     {%nt("adjunct_list_exclamative_cl")%}

adjunct_list_interrogative_cl ->
      adjunct            adjunct_list_interrogative_cl      {%nt("adjunct_list_interrogative_cl")%}
    | interrogative_cl   adjunct_list                       {%nt("adjunct_list_interrogative_cl")%}

adjunct_list_np ->
      adjunct adjunct_list_np                              {%nt("adjunct_list_np")%}
    | np      adjunct_list                                 {%nt("adjunct_list_np")%}

# can be out of order, not normally the case
# normal: i found the janitor happy 
# shifted: i found happy the janitor with two cars and a house
adjunct_list_np_ap -> 
      adjunct adjunct_list_np_ap            {%nt("adjunct_list_np_ap")%}
    | np      adjunct_list_ap               {%nt("adjunct_list_np_ap")%}
    | ap      adjunct_list_np               {%nt("adjunct_list_np_ap")%}

# can't shift: 
# - *i asked to eat the apple you
# on the other hand, adjuncts may sit between the np and the to inf
# - i asked you eariler to stop talking about this very long phrase 
adjunct_list_np_to_inf_cl ->                
      adjunct adjunct_list_np_to_inf_cl     {%nt("adjunct_list_np_ap")%}
    | np      adjunct_list_to_inf_cl        {%nt("adjunct_list_np_ap")%}

# bare inf can't shift
# - *i made stumble the huge giant that killed the last hero
# it seems like it can't even have a 
adjunct_list_np_bare_inf_cl ->
      adjunct adjunct_list_np_bare_inf_cl {%nt("adjunct_list_np_ap")%}
    | np      adjunct_list_bare_inf_cl    {%nt("adjunct_list_np_ap")%}

# shifting possible, if a to is used:
# - I told you earlier that we are out of grammars
# can't invert the declarative_cl and np tho:
# - *I told that I ate to the person that i saw earlier
# - *I told that I ate the person that i saw earlier
adjunct_list_np_declarative_cl -> 
      adjunct               adjunct_list_np_declarative_cl {%nt("adjunct_list_np_declarative_cl")%}
    | np                    adjunct_list_declarative_cl    {%nt("adjunct_list_np_declarative_cl")%}

# ditto
adjunct_list_np_exclamative_cl -> 
      adjunct               adjunct_list_np_exclamative_cl   {%nt("adjunct_list_np_exclamative_cl")%}
    | np                    adjunct_list_exclamative_cl      {%nt("adjunct_list_np_exclamative_cl")%}

# ditto
adjunct_list_np_interrogative_cl -> 
      adjunct                 adjunct_list_np_interrogative_cl {%nt("adjunct_list_np_interrogative_cl")%}
    | np                      adjunct_list_interrogative_cl    {%nt("adjunct_list_np_interrogative_cl")%}

# can invert the nps with a to
adjunct_list_np_np ->
      adjunct           adjunct_list_np_np                  {%nt("adjunct_list_np_np")%}
    | np                adjunct_list_np                     {%nt("adjunct_list_np_np")%}
    | to np             adjunct_list_np                     {%nt("adjunct_list_np_np")%}

fin_vp -> 
# modal
      advp? modal                   not? bare_inf_cl                           {%nt("fin_vp")%} # modal verb with bare infinitive clause argument (ex: "I can eat") 
# complete finite verb phrase
    | advp? vbf                     not? adjunct_list                          {%nt("fin_vp")%} # intransitive verb (ex: "I smoked")
    | advp? vbf_ap                  not? adjunct_list_ap                       {%nt("fin_vp")%} # intransitive verb with adjective phrase argument (ex: "You seemed happy")
    | advp? vbf_to_inf_cl           not? adjunct_list_to_inf_cl                {%nt("fin_vp")%} # intransitive verb with infinitive clause argument (ex: "I wanted to bring the book")
    | advp? vbf_bare_inf_cl         not? adjunct_list_bare_inf_cl              {%nt("fin_vp")%} # intransitive verb with bare infinitive clause argument (ex: "I helped clean")
    | advp? vbf_declarative_cl      not? adjunct_list_declarative_cl           {%nt("fin_vp")%} # intransitive verb with declarative content clause argument (ex: "I knew that you eat")
    | advp? vbf_exclamative_cl      not? adjunct_list_exclamative_cl           {%nt("fin_vp")%} # intransitive verb with exclamative content clause argument (ex: "I said how expensive it was.")
    | advp? vbf_interrogative_cl    not? adjunct_list_interrogative_cl         {%nt("fin_vp")%} # intransitive verb with interrogative clause argument (ex: "I knew what you eat")
    |       vbf_vbg_cl              not? vbg_cl                                {%nt("fin_vp")%} # past continuous (ex: "We were eating")
    |       vbf_vbn_cl              not? vbn_cl                                {%nt("fin_vp")%} # past perfect (ex: "He had eaten") OR passive voice (ex: "He was eaten")
    | advp? vbf_np                  not? adjunct_list_np                       {%nt("fin_vp")%} # transitive verb (ex: "I ate the apple")
    | advp? vbf_np_ap               not? adjunct_list_np_ap                    {%nt("fin_vp")%} # transitive verb with adjective phrase argument (ex: "I found you happy")
    | advp? vbf_np_to_inf_cl        not? adjunct_list_np_to_inf_cl             {%nt("fin_vp")%} # transitive verb with infinitive verb argument (ex: "I asked you to eat the apple")
    | advp? vbf_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl           {%nt("fin_vp")%} # transitive verb with bare infinitive verb argument (ex: "I made you eat the apple")
    | advp? vbf_np_declarative_cl   not? adjunct_list_np_declarative_cl        {%nt("fin_vp")%} # transitive verb with declarative content clause argument (ex: "I told you that you eat the apple")
    | advp? vbf_np_exclamative_cl   not? adjunct_list_np_exclamative_cl        {%nt("fin_vp")%} # transitive verb with exclamative content clause argument (ex: "I told you how expensive it was")
    | advp? vbf_np_interrogative_cl not? adjunct_list_np_interrogative_cl      {%nt("fin_vp")%} # transitive verb with interrogative clause argument (ex: "I asked you what you eat")
    | advp? vbf_np_np               not? adjunct_list_np_np                    {%nt("fin_vp")%} # ditransitive verb (ex: "I gave you food")

# a non-finite verb phrase
inf_vp ->
      advp? vb                      not? adjunct_list                          {%nt("inf_vp")%} # intransitive verb (ex: "to smoke")
    | advp? vb_ap                   not? adjunct_list_ap                       {%nt("inf_vp")%} # intransitive verb with adjective phrase argument (ex: "to seem happy")
    | advp? vb_to_inf_cl            not? adjunct_list_to_inf_cl                {%nt("inf_vp")%} # intransitive verb with infinitive clause argument (ex: "to want to eat")
    | advp? vb_bare_inf_cl          not? adjunct_list_bare_inf_cl              {%nt("inf_vp")%} # intransitive verb with bare infinitive clause argument (ex: "to help eat")
    | advp? vb_declarative_cl       not? adjunct_list_declarative_cl           {%nt("inf_vp")%} # intransitive verb with declarative content clause argument (ex: "to know that you eat")
    | advp? vb_exclamative_cl       not? adjunct_list_exclamative_cl           {%nt("inf_vp")%} # intransitive verb with exclamative content clause argument (ex: "to say how expensive it was.")
    | advp? vb_interrogative_cl     not? adjunct_list_interrogative_cl         {%nt("inf_vp")%} # intransitive verb with interrogative clause argument (ex: "to know what you eat")
    |       vb_vbg_cl               not?              vbg_cl                   {%nt("inf_vp")%} # present continuous (ex: "to be eating")
    |       vb_vbn_cl               not?              vbn_cl                   {%nt("inf_vp")%} # present perfect (ex: "to have eaten") OR passive voice (ex: "to be eaten")
    | advp? vb_np                   not? adjunct_list_np                       {%nt("inf_vp")%} # transitive verb (ex: "to eat food")    
    | advp? vb_np_ap                not? adjunct_list_np_ap                    {%nt("inf_vp")%} # transitive verb with adjective phrase argument (ex: "to find you happy")
    | advp? vb_np_to_inf_cl         not? adjunct_list_np_to_inf_cl             {%nt("inf_vp")%} # transitive verb with infinitive verb argument (ex: "to ask you to eat")
    | advp? vb_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl           {%nt("inf_vp")%} # transitive verb with bare infinitive verb argument (ex: "to make you eat")
    | advp? vb_np_declarative_cl    not? adjunct_list_np_declarative_cl        {%nt("inf_vp")%} # transitive verb with declarative content clause argument (ex: "to tell you that you eat")
    | advp? vb_np_exclamative_cl    not? adjunct_list_np_exclamative_cl        {%nt("inf_vp")%} # transitive verb with exclamative content clause argument (ex: "to tell you how expensive it was")
    | advp? vb_np_interrogative_cl  not? adjunct_list_np_interrogative_cl      {%nt("inf_vp")%} # transitive verb with interrogative clause argument (ex: "to ask you what you eat")
    | advp? vb_np_np                not? adjunct_list_np_np                    {%nt("inf_vp")%} # ditransitive verb (ex: "to give you food")


# present participle / gerund verb phrase
vbg_vp ->
      advp? vbg                     not? adjunct_list                          {%nt("vbg_vp")%}
    | advp? vbg_ap                  not? adjunct_list_ap                       {%nt("vbg_vp")%}
    | advp? vbg_to_inf_cl           not? adjunct_list_to_inf_cl                {%nt("vbg_vp")%}
    | advp? vbg_bare_inf_cl         not? adjunct_list_bare_inf_cl              {%nt("vbg_vp")%}
    | advp? vbg_declarative_cl      not? adjunct_list_declarative_cl           {%nt("vbg_vp")%}
    | advp? vbg_exclamative_cl      not? adjunct_list_exclamative_cl           {%nt("vbg_vp")%}
    | advp? vbg_interrogative_cl    not? adjunct_list_interrogative_cl         {%nt("vbg_vp")%}
    |       vbg_vbg_cl              not?              vbg_cl                   {%nt("vbg_vp")%}
    |       vbg_vbn_cl              not?              vbn_cl                   {%nt("vbg_vp")%}
    | advp? vbg_np                  not? adjunct_list_np                       {%nt("vbg_vp")%}
    | advp? vbg_np_ap               not? adjunct_list_np_ap                    {%nt("vbg_vp")%}
    | advp? vbg_np_to_inf_cl        not? adjunct_list_np_to_inf_cl             {%nt("vbg_vp")%}
    | advp? vbg_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl           {%nt("vbg_vp")%}
    | advp? vbg_np_declarative_cl   not? adjunct_list_np_declarative_cl        {%nt("vbg_vp")%}
    | advp? vbg_np_exclamative_cl   not? adjunct_list_np_exclamative_cl        {%nt("vbg_vp")%}
    | advp? vbg_np_interrogative_cl not? adjunct_list_np_interrogative_cl      {%nt("vbg_vp")%}
    | advp? vbg_np_np               not? adjunct_list_np_np                    {%nt("vbg_vp")%}

# a past participle verb phrase
vbn_vp ->
      advp? vbn                      not? adjunct_list                         {%nt("vbn_vp")%}
    | advp? vbn_ap                   not? adjunct_list_ap                      {%nt("vbn_vp")%}
    | advp? vbn_to_inf_cl            not? adjunct_list_to_inf_cl               {%nt("vbn_vp")%}
    | advp? vbn_bare_inf_cl          not? adjunct_list_bare_inf_cl             {%nt("vbn_vp")%}
    | advp? vbn_declarative_cl       not? adjunct_list_declarative_cl          {%nt("vbn_vp")%}
    | advp? vbn_exclamative_cl       not? adjunct_list_exclamative_cl          {%nt("vbn_vp")%}
    | advp? vbn_interrogative_cl     not? adjunct_list_interrogative_cl        {%nt("vbn_vp")%}
    |       vbn_vbg_cl               not?              vbg_cl                  {%nt("vbn_vp")%}
    |       vbn_vbn_cl               not?              vbn_cl                  {%nt("vbn_vp")%}
    | advp? vbn_np                   not? adjunct_list_np                      {%nt("vbn_vp")%}
    | advp? vbn_np_ap                not? adjunct_list_np_ap                   {%nt("vbn_vp")%}
    | advp? vbn_np_to_inf_cl         not? adjunct_list_np_to_inf_cl            {%nt("vbn_vp")%}
    | advp? vbn_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl          {%nt("vbn_vp")%}
    | advp? vbn_np_declarative_cl    not? adjunct_list_np_declarative_cl       {%nt("vbn_vp")%}
    | advp? vbn_np_exclamative_cl    not? adjunct_list_np_exclamative_cl       {%nt("vbn_vp")%}
    | advp? vbn_np_interrogative_cl  not? adjunct_list_np_interrogative_cl     {%nt("vbn_vp")%}
    | advp? vbn_np_np                not? adjunct_list_np_np                   {%nt("vbn_vp")%}


# versions of the adjunct_list with an np moved out

# following constituents are flat grammars that permit core arguments + some adjuncts in any order
adjunct_list_np_moved -> 
      adjunct           adjunct_list_np_moved  {%nt("adjunct_list_np_moved")%}
    | adjunct_np_moved  adjunct_list           {%nt("adjunct_list_np_moved")%}

adjunct_list_ap_np_moved ->
      adjunct     adjunct_list_ap_np_moved      {%nt("adjunct_list_ap_np_moved")%}
    | ap          adjunct_list_np_moved         {%nt("adjunct_list_ap_np_moved")%}
    | ap_np_moved adjunct_list                  {%nt("adjunct_list_ap_np_moved")%}

adjunct_list_to_inf_cl_np_moved ->
      adjunct              adjunct_list_to_inf_cl_np_moved  {%nt("adjunct_list_to_inf_cl_np_moved")%}
    | to_inf_cl            adjunct_list_np_moved            {%nt("adjunct_list_to_inf_cl_np_moved")%}
    | to_inf_cl_np_moved   adjunct_list                     {%nt("adjunct_list_to_inf_cl_np_moved")%}

adjunct_list_bare_inf_cl_np_moved ->
      bare_inf_cl_np_moved   adjunct_list           {%nt("adjunct_list_bare_inf_cl_np_moved")%}
    | bare_inf_cl            adjunct_list_np_moved  {%nt("adjunct_list_bare_inf_cl_np_moved")%}

adjunct_list_declarative_cl_np_moved ->
      adjunct                   adjunct_list_declarative_cl_np_moved      {%nt("adjunct_list_declarative_cl_np_moved")%}
    | adjunct_np_moved          adjunct_list_declarative_cl               {%nt("adjunct_list_declarative_cl_np_moved")%}
    | declarative_cl            adjunct_list_np_moved                     {%nt("adjunct_list_declarative_cl_np_moved")%}
    | declarative_cl_np_moved   adjunct_list                              {%nt("adjunct_list_declarative_cl_np_moved")%}

# allowables: exactly 1 np and 1 adjunct_np_moved, or all normal adjuncts
adjunct_list_np_np_moved ->
      adjunct           adjunct_list_np_np_moved               {%nt("adjunct_list_np_np_moved")%}
    | adjunct_np_moved  adjunct_list_np                        {%nt("adjunct_list_np_np_moved")%}
    | np                adjunct_list_np_moved                  {%nt("adjunct_list_np_np_moved")%}
    | null                                                     {%nt("adjunct_list_np_np_moved")%}

# should allow exactly 1 element where the np is moved, and the order of np and ap doesnt matter
adjunct_list_np_ap_np_moved -> 
      adjunct           adjunct_list_np_ap_np_moved            {%nt("adjunct_list_np_ap_np_moved")%}
    | adjunct_np_moved  adjunct_list_np_ap                     {%nt("adjunct_list_np_ap_np_moved")%}
    | np                adjunct_list_ap_np_moved               {%nt("adjunct_list_np_ap_np_moved")%}
    | ap                adjunct_list_np_np_moved               {%nt("adjunct_list_np_ap_np_moved")%}
    | ap_np_moved       adjunct_list_np                        {%nt("adjunct_list_np_ap_np_moved")%}

# should allow exactly 1 element where the np is moved, but np must preceede to_inf_cl
adjunct_list_np_to_inf_cl_np_moved ->                
      adjunct           adjunct_list_np_to_inf_cl_np_moved      {%nt("adjunct_list_np_to_inf_cl_np_moved")%}
    | adjunct_np_moved  adjunct_list_np_to_inf_cl               {%nt("adjunct_list_np_to_inf_cl_np_moved")%}
    | np                adjunct_list_to_inf_cl_np_moved         {%nt("adjunct_list_np_to_inf_cl_np_moved")%}
    | to_inf_cl         adjunct_list                            {%nt("adjunct_list_np_to_inf_cl_np_moved")%}

# TODO: may be too generous, should revise and ensure no bad grammar can slip past
adjunct_list_np_bare_inf_cl_np_moved ->                
      adjunct           adjunct_list_np_bare_inf_cl_np_moved      {%nt("adjunct_list_np_bare_inf_cl_np_moved")%}
    | adjunct_np_moved  adjunct_list_np_bare_inf_cl               {%nt("adjunct_list_np_bare_inf_cl_np_moved")%}
    | np                adjunct_list_bare_inf_cl_np_moved         {%nt("adjunct_list_np_bare_inf_cl_np_moved")%}
    | bare_inf_cl         adjunct_list                            {%nt("adjunct_list_np_bare_inf_cl_np_moved")%}

adjunct_list_np_declarative_cl_np_moved ->                
      adjunct                    adjunct_list_np_declarative_cl_np_moved      {%nt("adjunct_list_np_declarative_cl_np_moved")%}
    | adjunct_np_moved           adjunct_list_np_declarative_cl               {%nt("adjunct_list_np_declarative_cl_np_moved")%}
    | np                         adjunct_list_declarative_cl_np_moved         {%nt("adjunct_list_np_declarative_cl_np_moved")%}
    | declarative_cl             adjunct_list_np_np_moved                     {%nt("adjunct_list_np_declarative_cl_np_moved")%}
    | declarative_cl_np_moved    adjunct_list_np                              {%nt("adjunct_list_np_declarative_cl_np_moved")%}

# in exclamative and interrogative, we cannot mvoe out of the clause itself
adjunct_list_np_exclamative_cl_np_moved ->                
      adjunct                    adjunct_list_np_exclamative_cl_np_moved      {%nt("adjunct_list_np_exclamative_cl_np_moved")%}
    | adjunct_np_moved           adjunct_list_np_exclamative_cl               {%nt("adjunct_list_np_exclamative_cl_np_moved")%}
    | exclamative_cl             adjunct_list_np_np_moved                     {%nt("adjunct_list_np_exclamative_cl_np_moved")%}


adjunct_list_np_interrogative_cl_np_moved ->                
      adjunct                      adjunct_list_np_interrogative_cl_np_moved      {%nt("adjunct_list_np_interrogative_cl_np_moved")%}
    | adjunct_np_moved             adjunct_list_np_interrogative_cl               {%nt("adjunct_list_np_interrogative_cl_np_moved")%}
    | interrogative_cl             adjunct_list_np_np_moved                       {%nt("adjunct_list_np_interrogative_cl_np_moved")%}

# can invert the nps with a to
adjunct_list_np_np_np_moved ->
      adjunct           adjunct_list_np_np_np_moved                  {%nt("adjunct_list_np_np_np_moved")%}
    | adjunct_np_moved  adjunct_list_np_np                           {%nt("adjunct_list_np_np_np_moved")%}
    # indirect object specified
    | to np             adjunct_list_np_np_moved                     {%nt("adjunct_list_np_np_np_moved")%}
    # indirect object specified [ergo no dangling to]
    | np                adjunct_list_np_np_moved                     {%nt("adjunct_list_np_np_np_moved")%}
    # direct object specified [so there is a dangling to]
    | np                adjunct_list_np_do_moved                     {%nt("adjunct_list_np_np_np_moved")%}

# when the direct object is moved, yielding a dangling to
adjunct_list_np_do_moved -> 
      adjunct adjunct_list_np_do_moved                              {%nt("adjunct_list_np_do_moved")%}
    | to      adjunct_list                                          {%nt("adjunct_list_np_do_moved")%}

# a declarative clause but one of the noun phrases has been moved out of it
# this happens when a declarative clause is inside an interrogative clause
# this happens in sentences like "i know that you think that he eats the apple" -> "i know what you think that he eats"
# finite verb phrase with a noun phrase moved
fin_vp_np_moved ->
# modal
      advp? modal                    not? bare_inf_cl_np_moved                                        {%nt("fin_vp_np_moved")%} # modal verb with bare infinitive clause argument (ex: "I know what you [can eat]") 
# complete finite verb phrase
    | advp? vbf                      not? adjunct_list_np_moved                                       {%nt("fin_vp_np_moved")%} # I know who you [eat with]
    | advp? vbf_ap                   not? adjunct_list_ap_np_moved                                    {%nt("fin_vp_np_moved")%} # I know what you [seemed good at]
    | advp? vbf_to_inf_cl            not? adjunct_list_to_inf_cl_np_moved                             {%nt("fin_vp_np_moved")%} # I know what you [asked to bring]
    | advp? vbf_bare_inf_cl          not? adjunct_list_bare_inf_cl_np_moved                           {%nt("fin_vp_np_moved")%} # I know what you [helped bring]
    | advp? vbf_declarative_cl       not? adjunct_list_declarative_cl_np_moved                        {%nt("fin_vp_np_moved")%} # I know what you [said that you bring]
# cant move out of interrogative or exclamative clauses
    |       vbf_vbg_cl               not?              vbg_cl_np_moved                                {%nt("fin_vp_np_moved")%} # I know what you [were bringing]
    |       vbf_vbn_cl               not?              vbn_cl_np_moved                                {%nt("fin_vp_np_moved")%} # I know what you [had brought] OR I know what you [were brought]
    | advp? vbf_np                   not? adjunct_list_np_np_moved                                    {%nt("fin_vp_np_moved")%} # I know what you [brought]
    | advp? vbf_np_ap                not? adjunct_list_np_ap_np_moved                                 {%nt("fin_vp_np_moved")%} # I know what you [considered Bob good at]
    | advp? vbf_np_to_inf_cl         not? adjunct_list_np_to_inf_cl_np_moved                          {%nt("fin_vp_np_moved")%} # I know what you [asked Bob to bring]
    | advp? vbf_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl_np_moved                        {%nt("fin_vp_np_moved")%} # I know what you [helped Bob bring]
    | advp? vbf_np_declarative_cl    not? adjunct_list_np_declarative_cl_np_moved                     {%nt("fin_vp_np_moved")%} # I know what you [told Bob that you bring]
    | advp? vbf_np_exclamative_cl    not? adjunct_list_np_exclamative_cl_np_moved                     {%nt("fin_vp_np_moved")%} # I know who you [told how expensive it is]
    | advp? vbf_np_interrogative_cl  not? adjunct_list_np_interrogative_cl_np_moved                   {%nt("fin_vp_np_moved")%} # I know who you [asked what Bob brings]
    | advp? vbf_np_np                not? adjunct_list_np_np_np_moved                                 {%nt("fin_vp_np_moved")%} # I know what you [gave Bob]

# non-finite verb phrase with a noun phrase moved
inf_vp_np_moved ->
# complete infinitve verb phrase
      advp? vb                      not? adjunct_list_np_moved                                       {%nt("inf_vp_np_moved")%}
    | advp? vb_ap                   not? adjunct_list_ap_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vb_to_inf_cl            not? adjunct_list_to_inf_cl_np_moved                             {%nt("inf_vp_np_moved")%}
    | advp? vb_bare_inf_cl          not? adjunct_list_bare_inf_cl_np_moved                           {%nt("inf_vp_np_moved")%}
    | advp? vb_declarative_cl       not? adjunct_list_declarative_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    |       vb_vbg_cl               not?              vbg_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    |       vb_vbn_cl               not?              vbn_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    | advp? vb_np                   not? adjunct_list_np_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vb_np_ap                not? adjunct_list_np_ap_np_moved                                 {%nt("inf_vp_np_moved")%}
    | advp? vb_np_to_inf_cl         not? adjunct_list_np_to_inf_cl_np_moved                          {%nt("inf_vp_np_moved")%}
    | advp? vb_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    | advp? vb_np_declarative_cl    not? adjunct_list_np_declarative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vb_np_exclamative_cl    not? adjunct_list_np_exclamative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vb_np_interrogative_cl  not? adjunct_list_np_interrogative_cl_np_moved                   {%nt("inf_vp_np_moved")%}
    | advp? vb_np_np                not? adjunct_list_np_np_np_moved                                 {%nt("inf_vp_np_moved")%}

# non-finite verb phrase with a noun phrase moved
vbg_vp_np_moved ->
# complete infinitve verb phrase
      advp? vbg                      not? adjunct_list_np_moved                                       {%nt("inf_vp_np_moved")%}
    | advp? vbg_ap                   not? adjunct_list_ap_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vbg_to_inf_cl            not? adjunct_list_to_inf_cl_np_moved                             {%nt("inf_vp_np_moved")%}
    | advp? vbg_bare_inf_cl          not? adjunct_list_bare_inf_cl_np_moved                           {%nt("inf_vp_np_moved")%}
    | advp? vbg_declarative_cl       not? adjunct_list_declarative_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    |       vbg_vbg_cl               not?              vbg_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    |       vbg_vbn_cl               not?              vbn_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    | advp? vbg_np                   not? adjunct_list_np_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_ap                not? adjunct_list_np_ap_np_moved                                 {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_to_inf_cl         not? adjunct_list_np_to_inf_cl_np_moved                          {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_declarative_cl    not? adjunct_list_np_declarative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_exclamative_cl    not? adjunct_list_np_exclamative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_interrogative_cl  not? adjunct_list_np_interrogative_cl_np_moved                   {%nt("inf_vp_np_moved")%}
    | advp? vbg_np_np                not? adjunct_list_np_np_np_moved                                 {%nt("inf_vp_np_moved")%}

# non-finite verb phrase with a noun phrase moved
vbn_vp_np_moved ->
# complete infinitve verb phrase
      advp? vbn                      not? adjunct_list_np_moved                                       {%nt("inf_vp_np_moved")%}
    | advp? vbn_ap                   not? adjunct_list_ap_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vbn_to_inf_cl            not? adjunct_list_to_inf_cl_np_moved                             {%nt("inf_vp_np_moved")%}
    | advp? vbn_bare_inf_cl          not? adjunct_list_bare_inf_cl_np_moved                           {%nt("inf_vp_np_moved")%}
    | advp? vbn_declarative_cl       not? adjunct_list_declarative_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    |       vbn_vbg_cl               not?              vbg_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    |       vbn_vbn_cl               not?              vbn_cl_np_moved                                {%nt("inf_vp_np_moved")%}
    | advp? vbn_np                   not? adjunct_list_np_np_moved                                    {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_ap                not? adjunct_list_np_ap_np_moved                                 {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_to_inf_cl         not? adjunct_list_np_to_inf_cl_np_moved                          {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_bare_inf_cl       not? adjunct_list_np_bare_inf_cl_np_moved                        {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_declarative_cl    not? adjunct_list_np_declarative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_exclamative_cl    not? adjunct_list_np_exclamative_cl_np_moved                     {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_interrogative_cl  not? adjunct_list_np_interrogative_cl_np_moved                   {%nt("inf_vp_np_moved")%}
    | advp? vbn_np_np                not? adjunct_list_np_np_np_moved                                 {%nt("inf_vp_np_moved")%}



adjunct_list_ap_moved -> adjunct_list {%nt("adjunct_list_ap_moved")%}

adjunct_list_ap_ap_moved ->
      adjunct_list_ap    {%nt("adjunct_list_ap_ap_moved")%}
    | adjunct_list       {%nt("adjunct_list_ap_ap_moved")%}

adjunct_list_to_inf_cl_ap_moved ->
      adjunct_list_to_inf_cl                                {%nt("adjunct_list_to_inf_cl_ap_moved")%}
    | adjunct_list_to_inf_cl_ap_moved_                      {%nt("adjunct_list_to_inf_cl_ap_moved")%}

adjunct_list_to_inf_cl_ap_moved_ ->
      adjunct               adjunct_list_to_inf_cl_ap_moved_ {%nt("adjunct_list_to_inf_cl_ap_moved_")%}
    | to_inf_cl_ap_moved    adjunct_list                     {%nt("adjunct_list_to_inf_cl_ap_moved_")%}

adjunct_list_bare_inf_cl_ap_moved ->
      adjunct_list_bare_inf_cl                    {%nt("adjunct_list_bare_inf_cl_ap_moved")%} 
    | adjunct_list_bare_inf_cl_ap_moved_          {%nt("adjunct_list_bare_inf_cl_ap_moved")%} 

adjunct_list_bare_inf_cl_ap_moved_ ->
      bare_inf_cl_ap_moved   adjunct_list         {%nt("adjunct_list_bare_inf_cl_ap_moved_")%}

adjunct_list_declarative_cl_ap_moved ->
      adjunct_list_declarative_cl             {%nt("adjunct_list_declarative_cl_ap_moved")%}
    | adjunct_list_declarative_cl_ap_moved_   {%nt("adjunct_list_declarative_cl_ap_moved")%}

adjunct_list_declarative_cl_ap_moved_ ->
      adjunct                  adjunct_list_declarative_cl_ap_moved_    {%nt("adjunct_list_declarative_cl_ap_moved_")%}
    | declarative_cl_ap_moved  adjunct_list                             {%nt("adjunct_list_declarative_cl_ap_moved_")%}

adjunct_list_exclamative_cl_ap_moved -> adjunct_list_exclamative_cl {%nt("adjunct_list_exclamative_cl_ap_moved")%}

adjunct_list_interrogative_cl_ap_moved -> adjunct_list_interrogative_cl {%nt("adjunct_list_interrogative_cl_ap_moved")%}

adjunct_list_np_ap_moved -> adjunct_list_np {%nt("adjunct_list_np_ap_moved")%}

adjunct_list_np_ap_ap_moved -> 
      adjunct_list_np_ap            {%nt("adjunct_list_np_ap_ap_moved_")%}
    | adjunct_list_np               {%nt("adjunct_list_np_ap_ap_moved_")%}

adjunct_list_np_to_inf_cl_ap_moved ->
      adjunct_list_np_to_inf_cl                     {%nt("adjunct_list_np_to_inf_cl_ap_moved")%} 
    | adjunct_list_np_to_inf_cl_ap_moved_           {%nt("adjunct_list_np_to_inf_cl_ap_moved")%} 

adjunct_list_np_to_inf_cl_ap_moved_ -> 
      adjunct          adjunct_list_np_to_inf_cl_ap_moved_ {%nt("adjunct_list_np_to_inf_cl_ap_moved_")%}
    | np               adjunct_list_to_inf_cl_ap_moved_    {%nt("adjunct_list_np_to_inf_cl_ap_moved_")%}

adjunct_list_np_bare_inf_cl_ap_moved ->
      adjunct_list_np_bare_inf_cl                           {%nt("adjunct_list_np_bare_inf_cl_ap_moved")%} 
    | adjunct_list_np_bare_inf_cl_ap_moved_                 {%nt("adjunct_list_np_bare_inf_cl_ap_moved")%} 

adjunct_list_np_bare_inf_cl_ap_moved_ ->
      adjunct          adjunct_list_np_bare_inf_cl_ap_moved_   {%nt("adjunct_list_np_bare_inf_cl_ap_moved_")%}
    | np               adjunct_list_bare_inf_cl_ap_moved_      {%nt("adjunct_list_np_bare_inf_cl_ap_moved_")%}

adjunct_list_np_declarative_cl_ap_moved ->
      adjunct_list_np_declarative_cl                                {%nt("adjunct_list_np_declarative_cl_ap_moved")%}
    | adjunct_list_np_declarative_cl_ap_moved_                      {%nt("adjunct_list_np_declarative_cl_ap_moved")%}

adjunct_list_np_declarative_cl_ap_moved_ ->
      adjunct               adjunct_list_np_declarative_cl_ap_moved {%nt("adjunct_list_np_declarative_cl_ap_moved_")%}
    | np                    adjunct_list_declarative_cl_ap_moved    {%nt("adjunct_list_np_declarative_cl_ap_moved_")%}

adjunct_list_np_exclamative_cl_ap_moved -> 
      adjunct_list_np_exclamative_cl                         {%nt("adjunct_list_np_exclamative_cl_ap_moved")%}

adjunct_list_np_interrogative_cl_ap_moved -> 
      adjunct_list_np_interrogative_cl                         {%nt("adjunct_list_np_interrogative_cl_ap_moved")%}

adjunct_list_np_np_ap_moved ->
      adjunct_list_np_np           {%nt("adjunct_list_np_np_ap_moved")%}


fin_vp_ap_moved -> 
# modal
      advp? modal                   not? bare_inf_cl_ap_moved                           {%nt("fin_vp_ap_moved")%} # modal verb with bare infinitive clause argument (ex: "I know how happy you [can seem]") 
# finite verb phrase with a moved adjective phrase
    | advp? vbf                     not? adjunct_list_ap_moved                          {%nt("fin_vp_ap_moved")%} # intransitive verb (ex: "I know how much [you smoke]")
    | advp? vbf_ap                  not? adjunct_list_ap_ap_moved                       {%nt("fin_vp_ap_moved")%} # intransitive verb with adjective phrase argument (ex: "I know how happy you [seem]")
    | advp? vbf_to_inf_cl           not? adjunct_list_to_inf_cl_ap_moved                {%nt("fin_vp_ap_moved")%} # intransitive verb with infinitive clause argument (ex: "I know how happy you [want to seem]")
    | advp? vbf_bare_inf_cl         not? adjunct_list_bare_inf_cl_ap_moved              {%nt("fin_vp_ap_moved")%} # intransitive verb with bare infinitive clause argument (ex: "I know how well you [help clean]")
    | advp? vbf_declarative_cl      not? adjunct_list_declarative_cl_ap_moved           {%nt("fin_vp_ap_moved")%} # intransitive verb with declarative content clause argument (ex: "I know how happy you [know that you are]")
    | advp? vbf_exclamative_cl      not? adjunct_list_exclamative_cl_ap_moved           {%nt("fin_vp_ap_moved")%} # intransitive verb with exclamative content clause argument (ex: "I know how fast you [said how expensive it was]")
    | advp? vbf_interrogative_cl    not? adjunct_list_interrogative_cl_ap_moved         {%nt("fin_vp_ap_moved")%} # intransitive verb with interrogative clause argument (ex: "I know how quickly I [learned what you eat]")
    |       vbf_vbg_cl              not?              vbg_cl_ap_moved                   {%nt("fin_vp_ap_moved")%} # past continuous (ex: "I know how happy you [were seeming]")
    |       vbf_vbn_cl              not?              vbn_cl_ap_moved                   {%nt("fin_vp_ap_moved")%} # past perfect (ex: "I know how happy you [had seemed]") OR passive voice (ex: "I know how good bob [was considered]")
    | advp? vbf_np                  not? adjunct_list_np_ap_moved                       {%nt("fin_vp_ap_moved")%} # transitive verb (ex: "I know how happily you [ate the apple]")
    | advp? vbf_np_ap               not? adjunct_list_np_ap_ap_moved                    {%nt("fin_vp_ap_moved")%} # transitive verb with adjective phrase argument (ex: "I know how happy I [found you]")
    | advp? vbf_np_to_inf_cl        not? adjunct_list_np_to_inf_cl_ap_moved             {%nt("fin_vp_ap_moved")%} # transitive verb with infinitive verb argument (ex: "I know how skilled I [asked you to become]")
    | advp? vbf_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl_ap_moved           {%nt("fin_vp_ap_moved")%} # transitive verb with bare infinitive verb argument (ex: "I know how fast I [made you eat the apple]")
    | advp? vbf_np_declarative_cl   not? adjunct_list_np_declarative_cl_ap_moved        {%nt("fin_vp_ap_moved")%} # transitive verb with declarative content clause argument (ex: "I know how happy I [told you that you are]")
    | advp? vbf_np_exclamative_cl   not? adjunct_list_np_exclamative_cl_ap_moved        {%nt("fin_vp_ap_moved")%} # transitive verb with exclamative content clause argument (ex: "I know how quickly you [told me how expensive it was]")
    | advp? vbf_np_interrogative_cl not? adjunct_list_np_interrogative_cl_ap_moved      {%nt("fin_vp_ap_moved")%} # transitive verb with interrogative clause argument (ex: "I know how quickly I [asked you what you eat]")
    | advp? vbf_np_np               not? adjunct_list_np_np_ap_moved                    {%nt("fin_vp_ap_moved")%} # ditransitive verb (ex: "I know how quickly I [gave you food]")


inf_vp_ap_moved -> 
      advp? vb                     not? adjunct_list_ap_moved                          {%nt("inf_vp_ap_moved")%}
    | advp? vb_ap                  not? adjunct_list_ap_ap_moved                       {%nt("inf_vp_ap_moved")%}
    | advp? vb_to_inf_cl           not? adjunct_list_to_inf_cl_ap_moved                {%nt("inf_vp_ap_moved")%}
    | advp? vb_bare_inf_cl         not? adjunct_list_bare_inf_cl_ap_moved              {%nt("inf_vp_ap_moved")%}
    | advp? vb_declarative_cl      not? adjunct_list_declarative_cl_ap_moved           {%nt("inf_vp_ap_moved")%}
    | advp? vb_exclamative_cl      not? adjunct_list_exclamative_cl_ap_moved           {%nt("inf_vp_ap_moved")%}
    | advp? vb_interrogative_cl    not? adjunct_list_interrogative_cl_ap_moved         {%nt("inf_vp_ap_moved")%}
    |       vb_vbg_cl              not?              vbg_cl_ap_moved                   {%nt("inf_vp_ap_moved")%}
    |       vb_vbn_cl              not?              vbn_cl_ap_moved                   {%nt("inf_vp_ap_moved")%}
    | advp? vb_np                  not? adjunct_list_np_ap_moved                       {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_ap               not? adjunct_list_np_ap_ap_moved                    {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_to_inf_cl        not? adjunct_list_np_to_inf_cl_ap_moved             {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl_ap_moved           {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_declarative_cl   not? adjunct_list_np_declarative_cl_ap_moved        {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_exclamative_cl   not? adjunct_list_np_exclamative_cl_ap_moved        {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_interrogative_cl not? adjunct_list_np_interrogative_cl_ap_moved      {%nt("inf_vp_ap_moved")%}
    | advp? vb_np_np               not? adjunct_list_np_np_ap_moved                    {%nt("inf_vp_ap_moved")%}

vbn_vp_ap_moved -> 
      advp? vbn                     not? adjunct_list_ap_moved                          {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_ap                  not? adjunct_list_ap_ap_moved                       {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_to_inf_cl           not? adjunct_list_to_inf_cl_ap_moved                {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_bare_inf_cl         not? adjunct_list_bare_inf_cl_ap_moved              {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_declarative_cl      not? adjunct_list_declarative_cl_ap_moved           {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_exclamative_cl      not? adjunct_list_exclamative_cl_ap_moved           {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_interrogative_cl    not? adjunct_list_interrogative_cl_ap_moved         {%nt("vbn_vp_ap_moved")%}
    |       vbn_vbg_cl              not?              vbg_cl_ap_moved                   {%nt("vbn_vp_ap_moved")%}
    |       vbn_vbn_cl              not?              vbn_cl_ap_moved                   {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np                  not? adjunct_list_np_ap_moved                       {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_ap               not? adjunct_list_np_ap_ap_moved                    {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_to_inf_cl        not? adjunct_list_np_to_inf_cl_ap_moved             {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl_ap_moved           {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_declarative_cl   not? adjunct_list_np_declarative_cl_ap_moved        {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_exclamative_cl   not? adjunct_list_np_exclamative_cl_ap_moved        {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_interrogative_cl not? adjunct_list_np_interrogative_cl_ap_moved      {%nt("vbn_vp_ap_moved")%}
    | advp? vbn_np_np               not? adjunct_list_np_np_ap_moved                    {%nt("vbn_vp_ap_moved")%}

vbg_vp_ap_moved -> 
      advp? vbg                     not? adjunct_list_ap_moved                          {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_ap                  not? adjunct_list_ap_ap_moved                       {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_to_inf_cl           not? adjunct_list_to_inf_cl_ap_moved                {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_bare_inf_cl         not? adjunct_list_bare_inf_cl_ap_moved              {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_declarative_cl      not? adjunct_list_declarative_cl_ap_moved           {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_exclamative_cl      not? adjunct_list_exclamative_cl_ap_moved           {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_interrogative_cl    not? adjunct_list_interrogative_cl_ap_moved         {%nt("vbg_vp_ap_moved")%}
    |       vbg_vbg_cl              not?              vbg_cl_ap_moved                   {%nt("vbg_vp_ap_moved")%}
    |       vbg_vbn_cl              not?              vbn_cl_ap_moved                   {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np                  not? adjunct_list_np_ap_moved                       {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_ap               not? adjunct_list_np_ap_ap_moved                    {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_to_inf_cl        not? adjunct_list_np_to_inf_cl_ap_moved             {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_bare_inf_cl      not? adjunct_list_np_bare_inf_cl_ap_moved           {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_declarative_cl   not? adjunct_list_np_declarative_cl_ap_moved        {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_exclamative_cl   not? adjunct_list_np_exclamative_cl_ap_moved        {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_interrogative_cl not? adjunct_list_np_interrogative_cl_ap_moved      {%nt("vbg_vp_ap_moved")%}
    | advp? vbg_np_np               not? adjunct_list_np_np_ap_moved                    {%nt("vbg_vp_ap_moved")%}

# infintive clauses
to_inf_cl -> to inf_vp                   {%nt("to_inf_cl")%}
to_inf_cl_np_moved -> to inf_vp_np_moved {%nt("to_inf_cl_np_moved")%}
to_inf_cl_ap_moved -> to inf_vp_ap_moved {%nt("to_inf_cl_ap_moved")%}

bare_inf_cl -> inf_vp                     {%nt("bare_inf_cl")%}
bare_inf_cl_np_moved -> inf_vp_np_moved   {%nt("bare_inf_cl_np_moved")%}
bare_inf_cl_ap_moved -> inf_vp_ap_moved   {%nt("bare_inf_cl_ap_moved")%}

# vbg clause (participle)
vbg_cl -> vbg_vp                           {%nt("vbg_cl")%}
vbg_cl_np_moved -> vbg_vp_np_moved         {%nt("vbg_cl_np_moved")%}
vbg_cl_ap_moved -> vbg_vp_ap_moved         {%nt("vbg_cl_ap_moved")%}

# vbg clause (past participle)
vbn_cl -> vbn_vp                           {%nt("vbn_cl")%}
vbn_cl_np_moved -> vbn_vp_np_moved         {%nt("vbn_cl_np_moved")%}
vbn_cl_ap_moved -> vbn_vp_ap_moved         {%nt("vbn_cl_ap_moved")%}

# a declarative content clause
declarative_cl ->  that fin_cl                   {%nt("declarative_cl")%}

# an exclamative content clause
exclamative_cl -> 
      how advp fin_cl            {%nt("exclamative_cl")%} # how quickly mary became happy
    | how ap np fin_vp_ap_moved     {%nt("exclamative_cl")%} # how happy mary became

# an interrogative content clause
interrogative_cl -> 
      wh_np                      np fin_vp_np_moved            {%nt("interrogative_cl")%} # open interrogative clause with move from the core arguments
    | wh_np                         fin_vp                     {%nt("interrogative_cl")%} # open interrogative clause with move from the agent
    | wh_pp                      np fin_vp                     {%nt("interrogative_cl")%} # open interrogative clause with move from the adjuncts using pied piping (ex: "to where I go")
    | interrogative_subordinator np fin_vp                     {%nt("interrogative_cl")%} # closed interrogative clause (ex: "whether you eat the apple")

# a content clause with some np moved
declarative_cl_np_moved ->
      that    fin_vp              {%nt("declarative_cl_np_moved")%}
    | that np fin_vp_np_moved     {%nt("declarative_cl_np_moved")%}

# a content clause with some ap moved
declarative_cl_ap_moved ->
      that np fin_vp_ap_moved     {%nt("declarative_cl_ap_moved")%}
    | that np fin_vp              {%nt("declarative_cl_ap_moved")%}

np ->
    precorenp_modifier_list core_np postcorenp_modifier_list  {%nt("np")%}

precorenp_modifier_list -> precorenp_modifier:* {%nonterminal_unpack("precorenp_modifier_list")%} 

postcorenp_modifier_list -> postcorenp_modifier:* {%nonterminal_unpack("postcorenp_modifier_list")%}

# a noun phrase not including peripheral modifiers
core_np -> 
                    proper_noun                                     {%nt("core_np")%}  # a proper noun (ex: "John", "Mary")
    |               pronoun                                         {%nt("core_np")%}  # a pronoun (ex: "I", "you", "he", "she", "it", "we", "they")
    |               possessive_pronoun                              {%nt("core_np")%}  # a possessive pronoun (ex: "mine", "yours")
    |     ap_list   uncountable_noun n_modifier_list                {%nt("core_np")%}  # an uncountable noun with an adjective phrase (ex: "happy music") 
    | dp  ap_list   noun             n_modifier_list                {%nt("core_np")%}  # determiner phrase followed by a nominal (ex: "even all the lovely food too")


# a noun phrase that has been moved to the front (wh-movement)
wh_np -> wh    {%nt("wh_np")%} 
       | wh np {%nt("wh_np")%}

restrictive_correlative -> 
  that  {%nt("restrictive_correlative")%}
| which {%nt("restrictive_correlative")%}

restrictive_cl -> 
      restrictive_correlative    fin_vp              {%nt("restrictive_cl")%}
    | restrictive_correlative np fin_vp_np_moved     {%nt("restrictive_cl")%}
  
# a specifier coming after the noun
n_modifier -> restrictive_cl           {%nt("n_modifier")%} # a relative clause specifying the noun (ex: president who was elected) (ex: box that is on the table)
            | pp                       {%nt("n_modifier")%} # a prepositional phrase specifying the noun (ex: "the book on the table")

n_modifier_list -> n_modifier:* {%nonterminal_unpack("n_modifier_list")%}
# a determiner phrase
dp -> det                {%nt("det")%} # the, a, an, some, this, that
    | np s               {%nt("det")%} # a noun phrase followed by a possessive suffix (ex: "John's")
    | possessive_adjective {%nt("det")%} # a possessive pronoun (ex: "my", "your", "his", "her", "our", "their")

adjunct ->
      pp             {%nt("adjunct")%} # a prepositional phrase adjunct (ex: "in the house")
    | advp           {%nt("adjunct")%} # an adverb phrase adjunct (ex: "quickly")

adjunct_np_moved ->
      pp_np_moved   {%nt("adjunct")%}

# a prepositional phrase
pp ->   preposition np {%nt("pp")%}

# a prepositional phrase with np moved
pp_np_moved -> preposition {%nt("pp_np_moved")%}


# a prepositional phrase that has been moved to the front (pied piping)
wh_pp ->   preposition wh_np {%nt("wh_pp")%}

# an adjective phrase
ap ->
      advp                  ap              {%nt("ap")%} # an adverb phrase followed by an adjective phrase (ex: "very happy") 
    | adj                                   {%nt("ap")%} # an adjective (ex: "happy")
    | adj_pp                pp              {%nt("ap")%} # an adjective with a prepositional phrase argument (ex: "fond of music")
    | adj_declarative_cl    declarative_cl  {%nt("ap")%} # an adjective with a declarative content clause argument (ex: "happy that you are here")
    | adj_to_inf_cl         to_inf_cl       {%nt("ap")%} # an adjective with an infinitive clause argument (ex: "happy to be here")

ap_np_moved ->                                     
      advp ap_np_moved                              {%nt("ap_np_moved")%}
    | adj_pp             preposition                {%nt("ap_np_moved")%}
    | adj_declarative_cl declarative_cl_np_moved    {%nt("ap_np_moved")%}
    | adj_to_inf_cl      to_inf_cl_np_moved         {%nt("ap_np_moved")%}

# a sequence of aps
ap_list -> ap:* {%nonterminal_unpack("ap_list")%}

# an adverb phrase
advp -> adv             {%nt("advp")%} # an adverb (ex: "quickly")
      | adv advp        {%nt("advp")%} # an adverb phrase preceeded by another adverb phrase (ex: "very quickly")

advp? -> advp       {%nt("advp?")%}
       | null       {%nt("advp?")%}

not? -> not         {%nt("not?")%}
      | null        {%nt("not?")%}

# terminals

det -> %det {%t("det")%}
pronoun -> %pronoun {%t("pronoun")%}
possessive_adjective -> %possessive_adjective {%t("possessive_adjective")%}
possessive_pronoun -> %possessive_pronoun {%t("possessive_pronoun")%}
proper_noun -> %proper_noun {%t("proper_noun")%}
uncountable_noun -> %uncountable_noun {%t("uncountable_noun")%}
noun -> %noun {%t("noun")%}
preposition -> %preposition {%t("preposition")%}
to -> %to {%t("to")%}
s -> %s {%t("s")%}
not -> %not {%t("not")%}
that -> %that {%t("that")%}
interrogative_subordinator -> %interrogative_subordinator {%t("interrogative_subordinator")%}
modal -> %modal {%t("modal")%}
vb -> %vb {%t("vb")%}
vb_ap -> %vb_ap {%t("vb_ap")%}
vb_to_inf_cl -> %vb_to_inf_cl {%t("vb_to_inf_cl")%}
vb_bare_inf_cl -> %vb_bare_inf_cl {%t("vb_bare_inf_cl")%}
vb_declarative_cl -> %vb_declarative_cl {%t("vb_declarative_cl")%}
vb_exclamative_cl -> %vb_exclamative_cl {%t("vb_exclamative_cl")%}
vb_interrogative_cl -> %vb_interrogative_cl {%t("vb_interrogative_cl")%}
vb_vbg_cl -> %vb_vbg_cl {%t("vb_vbg_cl")%}
vb_vbn_cl -> %vb_vbn_cl {%t("vb_vbn_cl")%}
vb_np -> %vb_np {%t("vb_np")%}
vb_np_ap -> %vb_np_ap {%t("vb_np_ap")%}
vb_np_to_inf_cl -> %vb_np_to_inf_cl {%t("vb_np_to_inf_cl")%}
vb_np_bare_inf_cl -> %vb_np_bare_inf_cl {%t("vb_np_bare_inf_cl")%}
vb_np_declarative_cl -> %vb_np_declarative_cl {%t("vb_np_declarative_cl")%}
vb_np_exclamative_cl -> %vb_np_exclamative_cl {%t("vb_np_exclamative_cl")%}
vb_np_interrogative_cl -> %vb_np_interrogative_cl {%t("vb_np_interrogative_cl")%}
vb_np_np -> %vb_np_np {%t("vb_np_np")%}
vbg -> %vbg {%t("vbg")%}
vbg_ap -> %vbg_ap {%t("vbg_ap")%}
vbg_to_inf_cl -> %vbg_to_inf_cl {%t("vbg_to_inf_cl")%}
vbg_bare_inf_cl -> %vbg_bare_inf_cl {%t("vbg_bare_inf_cl")%}
vbg_declarative_cl -> %vbg_declarative_cl {%t("vbg_declarative_cl")%}
vbg_exclamative_cl -> %vbg_exclamative_cl {%t("vbg_exclamative_cl")%}
vbg_interrogative_cl -> %vbg_interrogative_cl {%t("vbg_interrogative_cl")%}
vbg_vbg_cl -> %vbg_vbg_cl {%t("vbg_vbg_cl")%}
vbg_vbn_cl -> %vbg_vbn_cl {%t("vbg_vbn_cl")%}
vbg_np -> %vbg_np {%t("vbg_np")%}
vbg_np_ap -> %vbg_np_ap {%t("vbg_np_ap")%}
vbg_np_to_inf_cl -> %vbg_np_to_inf_cl {%t("vbg_np_to_inf_cl")%}
vbg_np_bare_inf_cl -> %vbg_np_bare_inf_cl {%t("vbg_np_bare_inf_cl")%}
vbg_np_declarative_cl -> %vbg_np_declarative_cl {%t("vbg_np_declarative_cl")%}
vbg_np_exclamative_cl -> %vbg_np_exclamative_cl {%t("vbg_np_exclamative_cl")%}
vbg_np_interrogative_cl -> %vbg_np_interrogative_cl {%t("vbg_np_interrogative_cl")%}
vbg_np_np -> %vbg_np_np {%t("vbg_np_np")%}
vbn -> %vbn {%t("vbn")%}
vbn_ap -> %vbn_ap {%t("vbn_ap")%}
vbn_to_inf_cl -> %vbn_to_inf_cl {%t("vbn_to_inf_cl")%}
vbn_bare_inf_cl -> %vbn_bare_inf_cl {%t("vbn_bare_inf_cl")%}
vbn_declarative_cl -> %vbn_declarative_cl {%t("vbn_declarative_cl")%}
vbn_exclamative_cl -> %vbn_exclamative_cl {%t("vbn_exclamative_cl")%}
vbn_interrogative_cl -> %vbn_interrogative_cl {%t("vbn_interrogative_cl")%}
vbn_vbg_cl -> %vbn_vbg_cl {%t("vbn_vbg_cl")%}
vbn_vbn_cl -> %vbn_vbn_cl {%t("vbn_vbn_cl")%}
vbn_np -> %vbn_np {%t("vbn_np")%}
vbn_np_ap -> %vbn_np_ap {%t("vbn_np_ap")%}
vbn_np_to_inf_cl -> %vbn_np_to_inf_cl {%t("vbn_np_to_inf_cl")%}
vbn_np_bare_inf_cl -> %vbn_np_bare_inf_cl {%t("vbn_np_bare_inf_cl")%}
vbn_np_declarative_cl -> %vbn_np_declarative_cl {%t("vbn_np_declarative_cl")%}
vbn_np_exclamative_cl -> %vbn_np_exclamative_cl {%t("vbn_np_exclamative_cl")%}
vbn_np_interrogative_cl -> %vbn_np_interrogative_cl {%t("vbn_np_interrogative_cl")%}
vbn_np_np -> %vbn_np_np {%t("vbn_np_np")%}
vbf -> %vbf {%t("vbf")%}
vbf_ap -> %vbf_ap {%t("vbf_ap")%}
vbf_to_inf_cl -> %vbf_to_inf_cl {%t("vbf_to_inf_cl")%}
vbf_bare_inf_cl -> %vbf_bare_inf_cl {%t("vbf_bare_inf_cl")%}
vbf_declarative_cl -> %vbf_declarative_cl {%t("vbf_declarative_cl")%}
vbf_exclamative_cl -> %vbf_exclamative_cl {%t("vbf_exclamative_cl")%}
vbf_interrogative_cl -> %vbf_interrogative_cl {%t("vbf_interrogative_cl")%}
vbf_vbg_cl -> %vbf_vbg_cl {%t("vbf_vbg_cl")%}
vbf_vbn_cl -> %vbf_vbn_cl {%t("vbf_vbn_cl")%}
vbf_np -> %vbf_np {%t("vbf_np")%}
vbf_np_ap -> %vbf_np_ap {%t("vbf_np_ap")%}
vbf_np_to_inf_cl -> %vbf_np_to_inf_cl {%t("vbf_np_to_inf_cl")%}
vbf_np_bare_inf_cl -> %vbf_np_bare_inf_cl {%t("vbf_np_bare_inf_cl")%}
vbf_np_declarative_cl -> %vbf_np_declarative_cl {%t("vbf_np_declarative_cl")%}
vbf_np_exclamative_cl -> %vbf_np_exclamative_cl {%t("vbf_np_exclamative_cl")%}
vbf_np_interrogative_cl -> %vbf_np_interrogative_cl {%t("vbf_np_interrogative_cl")%}
vbf_np_np -> %vbf_np_np {%t("vbf_np_np")%}
adj -> %adj {%t("adj")%}
adj_pp -> %adj_pp {%t("adj_pp")%}
adj_declarative_cl -> %adj_declarative_cl {%t("adj_declarative_cl")%}
adj_to_inf_cl -> %adj_to_inf_cl {%t("adj_to_inf_cl")%}
adv -> %adv {%t("adv")%}
wh -> %wh {%t("wh")%}
why -> %why {%t("why")%}
how -> %how {%t("how")%}
which -> %which {%t("which")%}
precorenp_modifier -> %precorenp_modifier {%t("precorenp_modifier")%}
postcorenp_modifier -> %postcorenp_modifier {%t("postcorenp_modifier")%}
be_fin -> %be_fin {%t("is_fin")%}
do_fin -> %do_fin {%t("do_fin")%}
period -> %period {%t("period")%}
question_mark -> %question_mark {%t("question_mark")%}
exclamation_mark -> %exclamation_mark {%t("exclamation_mark")%}