"""
   Week 9 Lab: Classification and Feature Sets
   
   Goal: To learn how to set up a text classification problems in the NLTK
   
   Comments are added to the original .py file
   by Jenna Kim (as of 3/25/2018)

"""

''' --------------------------------------------------------------------

# Many NLP tasks such as WSD(Word Sense Disambiguation) and SRL (Semantic Role Labeling)
# can be treated as clssification problems and solved using machine learning techniques.
# To use NLP for classification, the input text needs to be represented 
# by a set of features for the classfier algorithm.
# In our labs, we will look at several ways to define text features for classification.

# In our first lab, we look at how to prepare data for classification in the NLTK.
# For each instance of the classification probelm, 
# we prepare a set of features 
# that will represent that instance to the machine learning algorithm.

# The below examples appear in Chapter 6 of the NLTK book.

''' 


import nltk

'''
### A. Name Gender Classifier

## Process of preparing text data for classification and training classifiers.

## In English, male and female first names have distinctive characteristics.
## For example, names ending in a, e, and i are likely to be female,
## while names ending in k, o, r, s and t are likely to be male.
## We will builde a classifier that will lable any name with its gender.

## Steps:
## Text instance -(feature extraction function)-> feature dictionary 
## feature dictionary -(make pair with correct label)-> feature set

'''                     

# 1. define a feature extraction function for each name
## This function generates a single feature which consists of the last letter of the name,
## returing a dictionary with a single item.

def gender_features(word):
    return{'last_letter': word[-1]}

print(gender_features('Shrek'))


## 2. Construct the training data, or "gold standard" data

## In this case, a list of first names, each of which will be labeled either male or female
## to construct the feature set for each name.
## For example, 'Shrek' - (gender_features function)-> {last_letter: 'k'}
## -(make pair with correct label)-> ({last_letter: 'k'}, 'male')

## NLTK corpus contains a names corpus that has a list of name first names
## and another list of female first names.
## We use this data to create a list of all the first names
## labeling each with its gender.


## if an error shows up while importing names
## nltk.download('names')

from nltk.corpus import names

## look at the first 20 male names
names.words('male.txt')[:20]

## look at the first 20 female names
names.words('female.txt')[:20]


## 2-a. From the name lists, create one long list with (name, gender) pairs to create the labeled data

namesgender = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])

## Take a look at the namesgender list
## how many names?		  
len(namesgender)

## first 20 names
namesgender[:20]

## last 20 names
namesgender[7924:]


## 2-b. To split the namesgender list into a training and test set,
## we first randomly suffle the names in the list.

import random

random.shuffle(namesgender)

## check out the first 20 names and compare them with the previous result.
namesgender[:20]

## 2-c. Use the feature extractor function (gender_features) to create the list of instances. 
## Featuresets represent each name as features and a gender label

featuresets = [(gender_features(n), g) for (n, g) in namesgender]

featuresets[:20]

## 2-d. Split the list into training and test sets:
## training - last 7444 examples; test - first 500 examples.

train_set, test_set = featuresets[500:], featuresets[:500]

## 2-e. And run the Naive Bayes classifier algorithm on training set
## to create a trained classifier

classifier = nltk.NaiveBayesClassifier.train(train_set)


## 2-f. Compute the accuracy of the classifier on the test set.
## Classify accuracy function first removes the gender labels from the test set
## and runs the classifier on each name in the test set to get a predicted gender. 
## Then, it compares the predicted gender with each actual gender from the test set
## to get the evaluation score.
## In this case, it just produces an accuracy score, instead of precision and recall.

print(nltk.classify.accuracy(classifier, test_set))

## 2-g. Use a classifier to label new instances, names that come from the future.

classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

## 2-h. Show the feature values that were most important in doing the classification.

classifier.show_most_informative_features(20)


## -----------------------------------------------------------------------


### B. Choosing Good Features

## Selecting relevant features can be important part training a classifier
## because using too many features with machine learning algorithms
## can cause an overfitting problem.
## The result is that the classifier is trained on so many of the exact details 
## of the training set that it does not work well on new examples.

## 1. Create a second feature extraction function
## that includes the first letter, the last letter, 
## a count of each letter, and the individual letters of the name.

def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

features = gender_features2('Shrek')
len(features)
features

## 2. Create feature sets using this function
featuresets2 = [(gender_features2(n), g) for (n, g) in namesgender]

## Take a look the featuresets2 by printing the the first 2 names as well 
for (n, g) in namesgender[:2]:
    print(n, gender_features2(n), '\n')

## 3. create new training and test sets base on these features,
## classify and look at accuracy

train_set, test_set = featuresets2[500:], featuresets2[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


## Note: the development process
## In developing a clssifier, it is necessary to do some error analysis of the test set
## Then perhaps change our features and retrain the classifier for better performance.
## So in the real world, our labeled data would be divided into a training, development, and test set.
## But for this lab, we continue to just use a training and a test set.


##--------------------------------------------------------------------------


### C. Experiment: Lab Exercise

## 1. Go back to using our features with just the last letter of each name.
## separate the names into training and test
## so we can repeat experiments on the same training and test set.

train_names = namesgender[500:]
test_names = namesgender[:500]

## 2.  use our original features to train a classifier and test on the development test set
train_set = [(gender_features(n), g) for (n, g) in train_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

### Save this classifier accuracy number 
### to use for comparison in the exercise.


## 3. Define a function that will compare the classifier labels with the gold standard labels

## The function will generate a list of error by running the classifier on the development test names
## and comparing it with the original name gender labels.

def geterrors(test):
    errors = []
    for (name, tag) in test:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )
    return errors

errors = geterrors(test_names)
len(errors)

## 4. Define a function to print all the errors, sorted by the correct labels
## so that we can look at the differences.

def printerrors(errors):
    for (tag, guess, name) in sorted(errors):
        print('correct={:<8s} guess={:<8s} name={:<30s}'.format(tag, guess, name))

printerrors(errors[:40])

## 5. Evaluation measures showing performance of classifier

## The confusion matrix shows the results of a test
## for how many of the actural class labels (gold standard labels)
## match with the predicted labels.
## In the NLTK, the confusion matrix is given by a function
## that takes two lists of labels for the test set.

from nltk.metrics import *

## 5-a. First, build the reference and test lists from the classifier on the test set
## reference list: all the correct/gold labels for the test set
## test list: all the predicted labels in the test set

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier.classify(features))

## Look at the first 30 labels in both lists	
reflist[:30]
testlist[:30]


## 5-b. using the NLTK function, define and print confusion matrix
cm = ConfusionMatrix(reflist, testlist)
print(cm)


## 6. Compute precision, recall and F-measure
## Precision = TP / (TP + FP)
## REcall = TP / (TP  + FN)
## F-measure = 2 * (recall * precision) / (recall + precision)

## To use the NLTK functions to get three scores,
## we need a different setup than the confusion matirx function
## which requires a set of item identifiers that are gold labels 
## and a set of item identifiers that are predicted labels

## 6-a. Set up a reference and test sets for each label
## that use the index number as the item identifiers 

reffemale = set()
refmale = set()
testfemale = set()
testmale = set()

for i, label in enumerate(reflist):
    if label == 'female': reffemale.add(i)
    if label == 'male': refmale.add(i)

for i, label in enumerate(testlist):
    if label == 'female': testfemale.add(i)
    if label == 'male': testmale.add(i)

reffemale
testfemale
refmale
testmale

## 6-b. Define a function that calls the 3 NLTK functions
## to get precision, recall and F-measure

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

## Print out the scores that show the model performance for each label

printmeasures('female', reffemale, testfemale)
printmeasures('male', refmale, testmale)


## ------------------------------------------------------------------------

### D. Your Own Exercise: 
## Make a post in the discussion for Week 9 lab in the Blackboard
## with your original accuracy on the test set you saved earlier 
## and the new accuracy from using another feature extraction function(below)
## You may also make any observations that you can about the remaining error.

## Define another feature extraction function for the exercise
## this function results two-letter suffixes
def gender_features3(word):
    return {'suffix1': word[-1],'suffix2': word[-2]}

gender_features3('Shrek')