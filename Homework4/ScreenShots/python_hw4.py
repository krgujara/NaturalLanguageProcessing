import nltk
import re
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentence_polarity
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from Subjectivity import *
from nltk.stem import PorterStemmer
from nltk.metrics import precision
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import ConfusionMatrix

nltk.download('averaged_perceptron_tagger')
nltk.download('sentence_polarity')
nltk.download('punkt')

mycorpus = PlaintextCorpusReader('.','clothing_shoes_jewelry.txt')
type(mycorpus.fileids())

file_id = mycorpus.fileids()
file_contents = []
file = file_id[0]
file_handle = open(file, 'r')
file_content = file_handle.read()
file_handle.close()
print(file_content[:500])

# extract all reviews

pword = re.compile('reviewerID:(.*)\nasin:(.*)\nreviewerName:(.*)\nhelpful:(.*)\nreviewText:(.*)\n(overall:.*)\nsummary:(.*)\nunixReviewTime:(.*)\nreviewTime:(.*)\n')
reviews = re.findall(pword,file_content)

len(reviews)

# extract the year
int(reviews[11][8][7:])

# reviews_2014 only first 5500 contains the reviews of year 2014
reviews_2014 = []
counter = 0
for i in range(len(reviews)):
    year = int(reviews[i][8][7:])
    if (year == 2014):
        # only [i][4] corresponds to the reviewText field. 
        reviews_2014.append(reviews[i][4])
        counter+=1
        if (counter >= 5500):
            break

len(reviews_2014)
reviews_2014[:2]

# Sentence level tokenization
sentences=[nltk.sent_tokenize(sent) for sent in reviews_2014]

sentences[-2]

# tokenize each review
# Sentence level tokenization, I am using Tweet Tokenizer because 
# if we use nltk.sent_tokenize, the words like didn't gets split, which we dont want.

tknzr = TweetTokenizer()
original_tokenized_review_sentences = []
for each_review in sentences:
    # tokenize each review into words by splitting them into different sentences
    sent_wordlevel=[tknzr.tokenize(sent) for sent in each_review]
    for each in sent_wordlevel:
        original_tokenized_review_sentences.append(each)
        
print(len(original_tokenized_review_sentences))
print(original_tokenized_review_sentences[-3:-1])

# converting the sentences to lower case to have the uniformity during classification
tokenized_review_sentences = []
# sentences now has the originally selected sentences from the reiviews file
# and lower_case_sentences has the 
for sentence in original_tokenized_review_sentences:
    tokenized_review_sentences.append([item.lower() for item in sentence])
    
print(tokenized_review_sentences[-3:-1])

print(len(tokenized_review_sentences))

# Stop Words
stop_words = set(stopwords.words('english'))
print(stop_words)

negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

neg_stop_words = []
print(type(stop_words))
for word in stop_words:
    if (word in negationwords) or (word.endswith("n't")):
        neg_stop_words.append(word)
    
#print(neg_stop_words)
#print(stop_words)      
neg_stop_words = set(neg_stop_words)
new_stop_words = []
new_stop_words = list(stop_words - neg_stop_words)


# Few stop words I thought will affect the classification, 
# like too, again 
# but as we can see in the below results, it really doesnt affect 
# the classification

print(new_stop_words)

list(swn.senti_synsets("too"))
breakdown3 = swn.senti_synset('besides.r.02')
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())

list(swn.senti_synsets("again"))
breakdown3 = swn.senti_synset('again.r.01')
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())

list(swn.senti_synsets("very"))
breakdown3 = swn.senti_synset('very.s.01')
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())


sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]


documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


featuresets = [(document_features(d,word_features), c) for (d,c) in documents]


train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


# Trying with 3000 most frequent bag of words
sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(3000)
word_features = [word for (word, freq) in word_items]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d,word_features), c) for (d,c) in documents]

train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


# recalculating Document Feature after removing stop words

sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

# all_word_list after removing the stop words
all_words_list = [word for (sent,cat) in documents for word in sent if word not in new_stop_words]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(3000)
word_features = [word for (word, freq) in word_items]

# bag of words approach
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# define the feature sets using the document_features
featuresets = [(document_features(d,word_features), c) for (d,c) in documents]

# Train and test your model for accuracy
train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))




SLpath = 'subjclueslen1-HLTEMNLP05.tff'
SL = readSubjectivity(SLpath)
print(SL['absolute'])

# define the features, to find out the feature_set
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        # count variables for the 4 classes of subjectivity
        weakPos = 0
        strongPos = 0
        weakNeg = 0
        strongNeg = 0
        for word in document_words:
            if word in SL:
                strength, posTag, isStemmed, polarity = SL[word]
                if strength == 'weaksubj' and polarity == 'positive':
                    weakPos += 1
                if strength == 'strongsubj' and polarity == 'positive':
                    strongPos += 1
                if strength == 'weaksubj' and polarity == 'negative':
                    weakNeg += 1
                if strength == 'strongsubj' and polarity == 'negative':
                    strongNeg += 1
                features['positivecount'] = weakPos + (2 * strongPos)
                features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features


#define the feature set for performinh the classification
# word features here is the revised word features after removing the stop words
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in documents]

print(SL_featuresets[0][0]['positivecount'])
print(SL_featuresets[0][0]['negativecount'])

train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))



negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features


# this word_features is the list of word_features after removing the stop words
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
NOT_featuresets[0][0]['contains(NOTlike)']
NOT_featuresets[0][0]['contains(always)']

train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(30)


def make_sentence(tokenized_review_sentence):
    sentence = ""
    for i in range(len(tokenized_review_sentence)):
        sentence+=  tokenized_review_sentence[i] + " "
    return sentence


positive_file_handle = open("positive.txt","w")
negative_file_handle = open("negative.txt","w")

for i in range(len(tokenized_review_sentences)):
    sent, orig_sent = tokenized_review_sentences[i], original_tokenized_review_sentences[i]
    if((classifier.classify(document_features(sent,word_features))) == 'pos'):
        positive_file_handle.write(make_sentence(orig_sent))
        positive_file_handle.write("\n")
    else:
        negative_file_handle.write(make_sentence(orig_sent))
        negative_file_handle.write("\n")

positive_file_handle.close()
negative_file_handle.close()


# Defining classifier, based on only adjectives and exclaimatory marks

sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

# all_word_list after removing the stop words
all_words_list = [word for (sent,cat) in documents for word in sent if word not in new_stop_words]

all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(15000)

# we can see that all_words_list currently has all the words like nouns
# adjectives, pronouns, exclaimation marks, proper nouns, etc
# print(word_items)

# Since we know that classification of sentiment is purely based on the adjectives
# negative words, etc

# lets remove all the special symbols from the list

print(type(word_items))
word_dict = dict(word_items)
print(type(word_dict))

# remove all non - alpha characters
pattern = "[^A-Za-z]+"
prog = re.compile(pattern)
refined_word_dict = dict()
for key in word_dict:
    if (prog.match(key)):
        # .,"--)?(:';!-2*[a]902002 all such words are removed from the most frequent words
        if (len(key)<=5):
            print("removing tuple with key "+ key)
        else:
            value = word_dict[key]
        refined_word_dict[key] = value

    else:
        value = word_dict[key]
        refined_word_dict[key] = value

# adding 2 important words externally to the word_dict
refined_word_dict['love'] = 100
refined_word_dict['hate'] = 100
        
print("Len After removing special symbols "+str(len(refined_word_dict)))
word_dict = dict()

# removing nouns, determinants from the list

for key in refined_word_dict:
    word_tokens = []
    word_tokens.append(key)
    tag = nltk.pos_tag(word_tokens) 
    tag = tag[0][1]
    if(tag == 'NN' or tag == 'NNS'):
        # print("removing Noun tuple with key "+ key)
        continue
    #elif (tag == 'VB' or tag == 'VBD' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ' ):
        # print("removing Verb tuple with key " + key)
    #    continue
    else:
        value = refined_word_dict[key]
        word_dict[key] = value
        # print(tag)
        
print("Len after removing nouns and verbs " + str(len(word_dict)))
# print(word_dict)
import copy

word_dict_copy = copy.copy(word_dict)

# stemming
ps = PorterStemmer()
for key in word_dict_copy:
    new_key = key
    new_key = ps.stem(new_key)
    new_value = word_dict_copy[key]
    word_dict[new_key] = new_value
    
# converting dictionary to list again
word_items = word_dict.items()
word_features = [word for (word, freq) in word_items]



# print(word_features)

SLpath = 'subjclueslen1-HLTEMNLP05.tff'
SL = readSubjectivity(SLpath)
print(SL['absolute'])

# define the features, to find out the feature_set
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        # count variables for the 4 classes of subjectivity
        weakPos = 0
        strongPos = 0
        weakNeg = 0
        strongNeg = 0
        for word in document_words:
            if word in SL:
                strength, posTag, isStemmed, polarity = SL[word]
                if strength == 'weaksubj' and polarity == 'positive':
                    weakPos += 1
                if strength == 'strongsubj' and polarity == 'positive':
                    strongPos += 1
                if strength == 'weaksubj' and polarity == 'negative':
                    weakNeg += 1
                if strength == 'strongsubj' and polarity == 'negative':
                    strongNeg += 1
                features['positivecount'] = weakPos + (2 * strongPos)
                features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features


#define the feature set for performinh the classification
# word features here is the revised word features after removing the stop words
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in documents]

print(SL_featuresets[0][0]['positivecount'])
print(SL_featuresets[0][0]['negativecount'])

train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label) 
    testlist.append(classifier.classify(features))


print(reflist[:30] )
print(testlist[:30])

dir(nltk.metrics)
from nltk.metrics import *


reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label) 
    testlist.append(classifier.classify(features))


cm = ConfusionMatrix(reflist, testlist)
print(cm)


from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
classifier=nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
nltk.classify.accuracy(classifier, test_set)


# calculating true negative, false positive, false negative, true positive
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(reflist, testlist).ravel()
(tn, fp, fn, tp)
