#!/usr/bin/env python

'compute metrics for libsvm test file and StreamSVM predictions file'

import sys
import numpy as np
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import confusion_matrix

y_file = sys.argv[1]
p_file = sys.argv[2]

print "loading p..."

p = np.loadtxt( p_file, usecols = [1] )

y_predicted = np.ones(( p.shape[0] ))
y_predicted[p < 0] = -1

print "loading y..."

y = np.loadtxt( y_file, usecols= [0] )

print "accuracy:", accuracy( y, y_predicted )
print "AUC:", AUC( y, p )

print
print "confusion matrix:"
print confusion_matrix( y, y_predicted )



