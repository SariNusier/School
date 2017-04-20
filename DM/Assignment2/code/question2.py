# --
# 7ccsmdm1 - lab9 solution
# sklar/19-mar-2017
# --
# THIS CODE IS AN ENTIRE COPY OF THE SOLUTION FOR LAB 9 MADE BY ELIZABETH SKLAR.
# I HAVE ADDED MY CODE TO INCLUDE THE CLASSIFICATION USING A TERM-DOCUMENT MATRIX.
# TO MAKE IT CLEAR, I HAVE WRITTEN A LIST OF THE LINES OF CODE WRITTEN BY ME BELOW.
# THE CODE FOR CREATING AND USING THE TERM-DOCUMENT MATRIX FOR CLASSIFICATION HAS BEEN INSPIRED FROM
# td-demo.py CODE INCLUDED WITH THIS COURSEWORK.

# LINES WRITTEN FOR DATA MINING - ASSIGNMENT 2:
# 23, 24: included other libraries
#

import sys
import nltk
import random
import string
import sklearn.model_selection as modsel
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_extraction import text
from sklearn import metrics

datafile = '../data/SMSSpamCollection.txt'

ratio = 0.3

# --PART 1: OBTAIN CORPUS AND STATS
try:
    with open(datafile, 'rt') as f:
        rawdata = f.readlines()
        rawrec = [rec for rec in rawdata]
except IOError as iox:
    print '** i/o error> ' + str(iox)
    sys.exit()

numrec = len(rawrec)
print 'number of records = %d' % (numrec)

X_data = []
y_target = []
numerr = 0
for rec in rawrec:
    try:
        rec2 = rec.split('\t')
        label = rec2[0]
        msg = rec2[1].strip()
        # do this to make sure there aren't any unreadable chars in the raw data
        words = nltk.word_tokenize(msg)
        X_data.append(msg)
        y_target.append(label)
    except Exception as x:
        numerr += 1
print 'number of input errors=%d' % (numerr)

# -split data set into training and test sets
X_train, X_test, y_train, y_test = modsel.train_test_split(X_data, y_target, test_size=0.50)

# --PART 2: CREATE BALANCED TRAINING SET AND A TEST SET

ham = []
spam = []
for i in range(len(X_train)):
    if (y_train[i] == 'ham'):
        ham.append(X_train[i])
    elif (y_train[i] == 'spam'):
        spam.append(X_train[i])
    else:
        numerr += 1  # count additional errors, if any
print 'number of: ham=%d spam=%d errors=%d total=%d' % (len(ham), len(spam), numerr, (len(ham) + len(spam) + numerr))

# add spam, if necessary
while (len(spam) < len(ham)):
    i = random.randint(0, len(spam) - 1)
    spam.append(spam[i])

# add ham, if necessary
while (len(ham) < len(spam)):
    i = random.randint(0, len(ham) - 1)
    ham.append(ham[i])

print 'balanced! number of: ham=%d spam=%d' % (len(ham), len(spam))

X_bal_train = []
y_bal_train = []
for j in range(int(len(ham))):
    i = random.randint(0, len(ham) - 1)
    X_bal_train.append(ham[i])
    y_bal_train.append('ham')
    del ham[i]
    i = random.randint(0, len(spam) - 1)
    X_bal_train.append(spam[i])
    y_bal_train.append('spam')
    del spam[i]

print 'size of data sets: training=%d test=%d' % (len(X_train), len(X_test))

# --PART 3: CHARACTERISE THE TRAINING SET

# attributes:
# msg_len
# num_digits
# num_upper
# num_punct
# num_words
num_attributes = 5
X = [[0 for i in range(num_attributes)] for j in range(len(X_bal_train))]
y = ['' for j in range(len(y_bal_train))]
for j in range(len(X_bal_train)):
    msg = X_bal_train[j]
    label = y_bal_train[j]
    # compute stats on message
    msg_len = len(msg)
    num_digits = 0
    num_upper = 0
    num_punct = 0
    for ch in msg:
        if (ch.isdigit()):
            num_digits += 1
        if (ch.isupper()):
            num_upper += 1
        if (ch in string.punctuation):
            num_punct += 1
    # split the message into words and compute stats on words
    words = nltk.word_tokenize(msg)
    num_words = len(words)
    # store label
    y[j] = label
    # store attributes in matrix
    X[j][0] = msg_len
    X[j][1] = num_digits
    X[j][2] = num_upper
    X[j][3] = num_punct
    X[j][4] = num_words

# -Initialise TfidfVectorizer() object
vectorizer = text.TfidfVectorizer()
# -Compute term-document matrix
td_train = vectorizer.fit_transform(X_train)
td_test = vectorizer.transform(X_test)

# --PART 4: TRAIN CLASSIFIER
# compare three classifiers:
# 1 = multinomial naive bayes
# 2 = averaging ensemble method: bagging
# 3 = boosting ensemble method: adaboost

# -start by instantiating an object for each classifier
clf = [MultinomialNB(), BaggingClassifier(), AdaBoostClassifier()]
clf_labels = ['Multinomial Bayes', 'Bagging', 'AdaBoost']
num_clf = len(clf)

clf_td = [MultinomialNB(), BaggingClassifier(), AdaBoostClassifier(), RandomForestClassifier()]
clf_labels_td = ['Multinomial Bayes', 'Bagging', 'AdaBoost', 'RandomForest']
num_clf_td = len(clf_td)

for c in range(num_clf_td):
    clf_td[c].fit(td_train, y_train)


for c in range(num_clf_td):
    pred = clf_td[c].predict(td_test)
    print clf_labels_td[c]
    confusion_matrix = metrics.confusion_matrix(y_test, pred, labels=['ham', 'spam'])
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    print 'TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN)
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    print 'precision=%f recall=%f f1=%f' % (precision, recall, f1)

"""
# -then fit a model for each classifier
for c in range(num_clf):
    clf[c].fit(X, y)

# -test how good they are
# store each statistic in a separate array, just to make the code clear
num_TP = [0 for i in range(3)]
num_TN = [0 for i in range(3)]
num_FP = [0 for i in range(3)]
num_FN = [0 for i in range(3)]
num_err = [0 for i in range(3)]

# -initialise array for storing attributes of test data instances
A = [0 for i in range(num_attributes)]

# -loop through test data
for j in range(len(X_test)):
    # grab message content and label for this test instance
    msg = X_test[j]
    label = y_test[j]
    # compute stats on message
    msg_len = len(msg)
    num_digits = 0
    num_upper = 0
    num_punct = 0
    for ch in msg:
        if (ch.isdigit()):
            num_digits += 1
        if (ch.isupper()):
            num_upper += 1
        if (ch in string.punctuation):
            num_punct += 1
    # split the message into words and compute stats on words
    words = nltk.word_tokenize(msg)
    num_words = len(words)
    # store attributes in vector
    A = np.array((msg_len, num_digits, num_upper, num_punct, num_words))
    # predict class with each classifier
    for c in range(num_clf):
        pred = clf[c].predict([A])
        # tally result: spam = positive, ham = negative
        if (label == 'spam'):
            if (pred == 'spam'):
                num_TP[c] += 1
            elif (pred == 'ham'):
                num_FN[c] += 1
            else:
                num_err[c] += 1
        elif (label == 'ham'):
            if (pred == 'spam'):
                num_FP[c] += 1
            elif (pred == 'ham'):
                num_TN[c] += 1
            else:
                num_err[c] += 1
        else:
            num_err[c] += 1

# -done testing classifiers on the test data.
# now compute and report statistics.

# -first, were there any data errors?
any_errors = -1
for c in range(num_clf):
    if (num_err[c] > 0):
        any_errors = c
if (any_errors > -1):
    print 'number of errors = %d' % (num_err[c])

# -then, print results for each classifier
for c in range(num_clf):
    print clf_labels[c]
    print 'TP=%d FP=%d TN=%d FN=%d' % (num_TP[c], num_FP[c], num_TN[c], num_FN[c])
    precision = num_TP[c] / float(num_TP[c] + num_FP[c])
    recall = num_TP[c] / float(num_TP[c] + num_FN[c])
    f1 = (2 * precision * recall) / (precision + recall)
    print 'precision=%f recall=%f f1=%f' % (precision, recall, f1)
"""