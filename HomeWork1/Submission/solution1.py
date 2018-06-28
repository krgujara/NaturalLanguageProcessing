import nltk
from nltk.corpus import PlaintextCorpusReader
mycorpus = PlaintextCorpusReader('.','.*.txt')
mycorpus.fileids()
part1 = mycorpus.fileids()[0]
part1
part1string = mycorpus.raw('state_union_part1.txt')
part1tokens = nltk.word_tokenize(part1string)
part1tokens[:100]
len(part1string)
len(part1tokens)
alphapart1 = [w for w in part1tokens if w.isalpha()]
alphapart1[:100]
alphalowerpart1 = [w.lower( ) for w in alphapart1]
alphalowerpart1[:50]
stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)
stopwords
stoppedalphalowerpart1 = [w for w in alphalowerpart1 if w not in stopwords]
from nltk import FreqDist
fdist = FreqDist(stoppedalphalowerpart1)
fdistkeys = list(fdist.keys())
fdistkeys[:50]
print('Printing top 50 words by frequency: ')
topkeys = fdist.most_common(50)
for pair in topkeys:
    print(pair)
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(alphalowerpart1)
scored = finder.score_ngrams(bigram_measures.raw_freq)


type(scored)
first = scored[0]
type(first)
first
print('Printing Top 50 Bigrams by frequency without applying stop word filter')
for bscore in scored[:50]:
    print (bscore)
stopwords.append("united")
stopwords.append("states")


finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)

print('Printing Top 50 Bigrams by frequency after applying stop word filter')
for bscore in scored[:50]:
    print (bscore)


print('Printing Top 50 Bigrams by frequency after applying stop word filter, and less than equal to 3 filter')
finder.apply_ngram_filter(lambda w1, w2: len(w1) < 4)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)

print('Printing top 50 bigrams by their Mutual Information Scores (using min freq 5)')
finder2 = BigramCollocationFinder.from_words(stoppedalphalowerpart1)
finder2.apply_freq_filter(5)
scored = finder2.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)


