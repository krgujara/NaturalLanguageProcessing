import nltk
from nltk.corpus import PlaintextCorpusReader
mycorpus = PlaintextCorpusReader('.','.*.txt')
mycorpus.fileids()
part2 = mycorpus.fileids()[1]
part2
part2string = mycorpus.raw('state_union_part2.txt')
part2tokens = nltk.word_tokenize(part2string)
part2tokens[:100]

len(part2string)
len(part2tokens)
alphapart2 = [w for w in part2tokens if w.isalpha()]
alphapart2[:100]
alphalowerpart2 = [w.lower( ) for w in alphapart2]
alphalowerpart2[:50]
stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)
stopwords
stoppedalphalowerpart2 = [w for w in alphalowerpart2 if w not in stopwords]
from nltk import FreqDist
fdist = FreqDist(stoppedalphalowerpart2)
fdistkeys = list(fdist.keys())
fdistkeys[:50]
print('Printing top 50 words by frequency: ')
topkeys = fdist.most_common(50)
for pair in topkeys:
    print(pair)
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(alphalowerpart2)
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
print('Printing Top 50 Bigrams by frequency after applying stop word filter')
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)

print('Printing Top 50 Bigrams by frequency after applying stop word filter, and less than equal to 3 filter')
finder.apply_ngram_filter(lambda w1, w2: len(w1) < 4)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)

finder2 = BigramCollocationFinder.from_words(stoppedalphalowerpart2)
finder2.apply_freq_filter(5)

print('Printing top 50 bigrams by their Mutual Information Scores (using min freq 5)')
scored = finder2.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)






