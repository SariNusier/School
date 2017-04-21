# --
# sd-clean.py
# sklar/28-mar-2017
#
# This code cleans the Speed Dating data set to extract the features
# we are interested in.
# --

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics
from sklearn import decomposition
from sklearn import preprocessing

try:
    # open data file in csv format
    f = open('../data/speed_dating.csv', 'rU')
    # read contents of data file into "rawdata" list
    indata = csv.reader(f)
    # parse data in csv format
    rawdata = [rec for rec in indata]
# handle exceptions:
except IOError as iox:
    print '** I/O error trying to open the data file> ' + str(iox)
    sys.exit()
except Exception as x:
    print '** error> ' + str(x)
    sys.exit()

# -handle header in CSV file
# first record should be header: print it and delete it from rawdata list
header = rawdata[0]
del rawdata[0]

# -select fields of interest:
#  i=0 : unique subject ID number
#  i=2 : gender (female=0, male=1)
#  i=9 : order
# i=11 : partner's unique subject ID number
# i=12 : match (0=no, 1=yes)
# i=13 : correlation between subject's and partner's ratings of interests
# i=15 : age of partner
# i=16 : race of partner
# i=33 : age of subject
# i=39 : race of subject
# i=46 : how frequently do you go on dates
# i=47 : how frequently do you go out
features = [0, 2, 9, 11, 13, 15, 16, 33, 39, 46, 47]
num_features = len(features)
label = 12

# -gather fields of interest from full data set into X and y
X = []
y = []
num_err = 0
for rec in rawdata:
    try:
        instance = []
        for f in features:
            instance.append(float(rec[f]))
        X.append(instance)
        y.append(rec[label])
    except Exception as x:
        num_err += 1
        # print '** error> ' + str( x ) + ', feature index=' + str( f ) + ',
        # print ' feature=[',
        # print rec[f],
        # print '], raw data=['
        # print rec
        # print ']'
print 'number of errors = %d' % num_err
X = np.array(X)
y = np.array(y)
print y
num_instances = X.shape[0]
print 'number of instances = %d' % num_instances
print 'shape of input data = %d x %d' % (X.shape[0], X.shape[1])
print 'shape of target data = %d' % (y.shape[0])

# === MY CODE STARTS HERE ===

n_clusters = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
normalized = preprocessing.normalize(X)
clst = [cluster.KMeans(n_clusters=c_no).fit(normalized) for c_no in n_clusters]

pcas = [decomposition.PCA(n_components=n_comp) for n_comp in range(2,10)]

# for c in clst:
    # print c.inertia_
    # print metrics.silhouette_score(normalized, c.labels_)
    # print metrics.calinski_harabaz_score(normalized, c.labels_)

for p in pcas:
    data = p.fit_transform(normalized)
    p_clst = [cluster.KMeans(n_clusters=c_no).fit(data) for c_no in n_clusters]
    for c in p_clst:
        print c.inertia_
    print
