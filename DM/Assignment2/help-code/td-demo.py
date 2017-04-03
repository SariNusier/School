#--
# td-demo.py
# sklar/27-mar-2017
#
# based on http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset
#
# This program demonstrates use of the functions in scikit-learn that
# generate a term-document matrix and perform classification using
# that matrix.
#
# The code uses the sklearn.datasets.fetch_20newsgroups class, which
# contains data fetching/caching functions that download the data
# archive from the original 20 newsgroups website
# (http://qwone.com/~jason/20Newsgroups/).
#
#--

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import scipy
from pprint import pprint


#-5.8.1. Get text data
# Download data from the 20 newsgroups website, extract the archive
# contents into a directory called "scikit_learn_data/20news_home" and
# call sklearn.datasets.load_files.
# Here we fetch newsgroup names for only some categories (i.e., those
# that match the "cats" list; use None to fetch data for all
# newsgroups).
cats = [ 'comp.graphics', 'sci.space', 'misc.forsale' ]
newsgroups_train = fetch_20newsgroups( data_home='.', subset='train', categories=cats, download_if_missing=True )
# Print some useful information to help you understand what was
# fetched...
print 'target names from fetch_20newsgroups (for categories ',
print cats,
print ')=',
pprint( list( newsgroups_train.target_names ))
print 'number of instances (newsgroups_train.filenames.shape) =',
print newsgroups_train.filenames.shape[0]
print 'number of targets (newsgroups_train.target.shape) =',
print newsgroups_train.target.shape[0]
# target values are indexes into the "cats" list
print 'first 50 target values (newsgroups_train.target[:50]) =',
print newsgroups_train.target[:50]


#-5.8.2. Convert text to vectors
# Create a term-document matrix across all the training data, where
# each instance in the training set is a row and each column is a term
# from the full data set.
# For more information, see
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

#-Initialise TfidfVectorizer() object
vectorizer = text.TfidfVectorizer()
#-Compute term-document matrix
td_train = vectorizer.fit_transform( newsgroups_train.data )
#-Print some useful information to help you understand what was
#-computed...
print 'number of documents (vectors.shape[0]) =',
print td_train.shape[0]
print 'number of terms (vectors.shape[1]) =',
print td_train.shape[1]

#-Note that the term-document matrix is generally quite sparse. We can
# print some information about that. Note that the term-document
# matrix is stored as a scipy.sparse.csr_matrix data structure.
# For more information on this type of data structure, see
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
print 'number of explicitly-stored values (non-zeros) in term-document matrix = ',
print td_train.nnz
print 'sparseness of term-document matrix =',
print td_train.nnz / float( td_train.shape[0] * td_train.shape[1] )


#-5.8.3. Classify!

#-Classify training data
clf = MultinomialNB( alpha=.01 )
clf.fit( td_train, newsgroups_train.target )

#-Grab test set
newsgroups_test = fetch_20newsgroups( subset='test', categories=cats )

#-Compute term-document matrix on test set
td_test = vectorizer.transform( newsgroups_test.data )

#-Run prediction based on test set
pred = clf.predict( td_test )

#-Report metrics
print 'f1 score = %f' % metrics.f1_score( newsgroups_test.target, pred, average='macro' )
