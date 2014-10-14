#!/usr/bin/env python

'compute metrics for libsvm test file and VW/Liblinear predictions file'

import sys
import numpy as np
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import confusion_matrix

y_file = sys.argv[1]
p_file = sys.argv[2]

print "loading p..."

p = np.loadtxt( p_file )

y_predicted = np.ones(( p.shape[0] ))
y_predicted[p < 0] = -1

print "loading y..."

y = np.loadtxt( y_file, usecols= [0] )

print "accuracy:", accuracy( y, y_predicted )
print "AUC:", AUC( y, p )

print
print "confusion matrix:"
print confusion_matrix( y, y_predicted )


"""
run score.py data/test_v.txt vw/p_v_logistic.txt

accuracy: 0.994675826535

confusion matrix:
[[27444   136]
 [  236 42054]]

AUC: 0.998418419401
"""

"""
p_v_hinge.txt

accuracy: 0.993502218406

confusion matrix:
[[27310   270]
 [  184 42106]]

AUC: 0.99632599445
"""

"""
cdblock

accuracy: 0.993244597109
AUC: 0.993511427279

confusion matrix:
[[27436   144]
 [  328 41962]]
"""

"""
cdblock -s 7 (logistic regression)
accuracy: 0.985201087734
AUC: 0.985763288671

confusion matrix:
[[27261   319]
 [  715 41575]]
"""

"""
score_streamsvm.py (hinge)

accuracy: 0.990596822671
AUC: 0.991292619197

confusion matrix:
[[27431   149]
 [  508 41782]]
 

(ui) 
accuracy: 0.990596822671
AUC: 0.998972438313

confusion matrix:
[[27431   149]
 [  508 41782]]
 
"""