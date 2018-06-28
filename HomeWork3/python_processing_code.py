# Context Free Grammar to parse 4 given sentences

import nltk
from nltk import FreqDist

grammar = nltk.CFG.fromstring("""
S -> NP VP | VP
NP -> PRP | DT ADJP NN | NN | PRP NNS | CD NNS
VP -> VBD NP NP | MD VP | VB ADVP | VBP RB ADVP ADJP | VBD S | TO VP | VB NP ADVP
ADVP -> RB | NP RB
NNS -> "kids" | "days"
RB -> "now" | "always" | "not" | "ago"
VB -> "go" | "visit"
MD -> "may"
ADJP -> JJ 
PRP -> "We" |"You" | "Their" | "She" | "me"
VBD -> "had" | "came"
CD -> "two"
VBP -> "are"
DT -> "a"
TO -> "to"
JJ -> "nice" | "naive"
NN -> "party" | "yesterday"
""")


# parsing first sentence - "We had a nice party yesterday"
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "We had a nice party yesterday"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)




# parsing second sentence - "She came to visit me two days ago
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "She came to visit me two days ago"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)


# parsing third sentence - "You may go now"
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "You may go now"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)



# parsing fourth sentence - "Their kids are not always naive"
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "Their kids are not always naive"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)


# sample sentence 1
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "She may visit now"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)

# sample sentence 2
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "You are not always nice"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)


# sample sentence 3
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "She came two kids yesterday"
sentlist = senttext.split()
print(sentlist)

trees = rd_parser.parse(sentlist)
trees
treelist = list(trees)

type(treelist[0]) 
for tree in treelist:
    print (tree)


# Creating a mini corpus of 4 given sentences to get the probablistic 
# frequency of each given word
corpus = "We had a nice party yesterday She came to visit me two days ago You may go now Their kids are not always naive"
corpus_words = corpus.split()
fdist = FreqDist(corpus_words)
# fdist for each pair is 1 which means that each word occurs with the equal probablity
fdist



# Probablistic Grammar
# The probabilities for each non-terminal symbol must add up to 1

prob_grammar = nltk.PCFG.fromstring("""
S -> NP VP[0.9] | VP [0.1]
NP -> PRP [0.5]| DT ADJP NN [0.2]| NN [0.1]| PRP NNS [0.1]| CD NNS[0.1]
VP -> VBD NP NP [0.3]| MD VP [0.2]| VB ADVP[0.1] | VBP RB ADVP ADJP[0.1] | VBD S [0.1]| TO VP[0.1] | VB NP ADVP[0.1]
ADVP -> RB [0.5]| NP RB[0.5]
NNS -> "kids"[0.5] | "days"[0.5]
RB -> "now"[0.25] | "always"[0.25] | "not"[0.25] | "ago"[0.25]
VB -> "go"[0.5] | "visit"[0.5]
MD -> "may"[1.0]
ADJP -> JJ [1.0]
PRP -> "We" [0.2]|"You"[0.2] | "Their"[0.2] | "She"[0.2] | "me"[0.2]
VBD -> "had"[0.5] | "came"[0.5]
CD -> "two"[1.0]
VBP -> "are"[1.0]
DT -> "a"[1.0]
TO -> "to"[1.0]
JJ -> "nice" [0.5]| "naive"[0.5]
NN -> "party" [0.5]| "yesterday"[0.5]
""")

viterbi_parser = nltk.ViterbiParser(prob_grammar)

for tree in viterbi_parser.parse(['We' ,'had','a', 'nice', 'party', 'yesterday']):
    print (tree)

for tree in viterbi_parser.parse(['She' ,'came', 'to', 'visit', 'me' ,'two' ,'days', 'ago']):
    print (tree)

for tree in viterbi_parser.parse(['You' ,'may', 'go' ,'now']):
    print (tree)

for tree in viterbi_parser.parse(['Their', 'kids', 'are', 'not' ,'always', 'naive']):
    print (tree)