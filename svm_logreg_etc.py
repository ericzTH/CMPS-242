"""
vectorizer toarray shape: (4743, 12230)
 after pca: (4743, 300)
test data shape after TruncatedSVD: (1701, 300)
_________________________________________________________________________________________
Regularizers:   |LogReg Training Accuracies:    |       LogReg Validation Accuracies:
_________________________________________________________________________________________
        1e-05   |       0.593641800441          |       0.563856960409
        0.0001  |       0.605288007554          |       0.575989782886
        0.001   |       0.693421466793          |       0.663473818646
        0.01    |       0.857097891092          |       0.840996168582
        0.1     |       0.88825936418           |       0.871008939974
        1       |       0.91532892666           |       0.891443167305
        10      |       0.93452943028           |       0.909961685824
        100     |       0.947749449166          |       0.901660280971
        1000    |       0.953415171545          |       0.897190293742
        10000   |       0.955303745672          |       0.893997445722
_________________________________________________________________________________________
rbf kernel
___________________________________________________________________________________
Regularizers:   |SVM Training Accuracies:       |       SVM Validation Accuracies:
___________________________________________________________________________________
        99999.99|       0.502675480013          |       0.488505747126
        10000.0 |       0.502675480013          |       0.488505747126
        1000.0  |       0.502675480013          |       0.488505747126
        100.0   |       0.502675480013          |       0.488505747126
        10.0    |       0.502675480013          |       0.488505747126
        1.0     |       0.502675480013          |       0.488505747126
        0.1     |       0.874409820585          |       0.854406130268
        0.01    |       0.916273213724          |       0.904853128991
        0.001   |       0.937047529116          |       0.913154533844
        0.0001  |       0.9537299339            |       0.899106002554
___________________________________________________________________________________
svm linear
___________________________________________________________________________________
Regularizers:   |SVM Training Accuracies:       |       SVM Validation Accuracies:
___________________________________________________________________________________
        99999.99|       0.502675480013          |       0.488505747126
        10000.0 |       0.502675480013          |       0.488505747126
        1000.0  |       0.502675480013          |       0.488505747126
        100.0   |       0.502675480013          |       0.488505747126
        10.0    |       0.894239848914          |       0.874201787995
        1.0     |       0.918476550205          |       0.90932311622
        0.1     |       0.938621340888          |       0.906130268199
        0.01    |       0.953415171545          |       0.897190293742
        0.001   |       0.958136606862          |       0.889527458493
        0.0001  |       0.958451369216          |       0.886334610473
___________________________________________________________________________________
svm poly
___________________________________________________________________________________
Regularizers:   |SVM Training Accuracies:       |       SVM Validation Accuracies:
___________________________________________________________________________________
        99999.99|       0.502675480013          |       0.488505747126
        10000.0 |       0.502675480013          |       0.488505747126
        1000.0  |       0.502675480013          |       0.488505747126
        100.0   |       0.502675480013          |       0.488505747126
        10.0    |       0.502675480013          |       0.488505747126
        1.0     |       0.502675480013          |       0.488505747126
        0.1     |       0.502675480013          |       0.488505747126
        0.01    |       0.502675480013          |       0.488505747126
        0.001   |       0.502675480013          |       0.488505747126
        0.0001  |       0.502675480013          |       0.488505747126
___________________________________________________________________________________
svm sigmoid
___________________________________________________________________________________
Regularizers:   |SVM Training Accuracies:       |       SVM Validation Accuracies:
___________________________________________________________________________________
        99999.99|       0.502675480013          |       0.488505747126
        10000.0 |       0.502675480013          |       0.488505747126
        1000.0  |       0.502675480013          |       0.488505747126
        100.0   |       0.502675480013          |       0.488505747126
        10.0    |       0.502675480013          |       0.488505747126
        1.0     |       0.502675480013          |       0.488505747126
        0.1     |       0.757947749449          |       0.719029374202
        0.01    |       0.909977966635          |       0.893997445722
        0.001   |       0.931381806736          |       0.911877394636
        0.0001  |       0.946175637394          |       0.902298850575
___________________________________________________________________________________
"""
import warnings,math
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
import pylab as pl
import tensorflow as tf
import operator,time,sys,os
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn import linear_model,metrics
from sklearn.mixture import GMM
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV,train_test_split,KFold, cross_val_score
import sklearn,matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def load(train,test,labels):

	if os.name != 'posix':
		os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
	train_data = pd.read_csv(train,header = None)
	labels_train = pd.read_csv(labels,header = None)
	labels_train[0] = labels_train[0].map({'HC':1,'DT':0})
	test_data = pd.read_csv(test)
	return(train_data,test_data,labels_train[0])

train,test,lables = load("train.csv","test.csv","labels_train_tweets.csv")

text = train.copy()
test_text = test['tweet'].copy()


# tf-idf on train data
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase = False)
#temp_x = vectorizer.fit_transform(text)
mat_vectorizer = vectorizer.fit_transform(text[0]).toarray()
pca = PCA(n_components = 300)
mat = pca.fit_transform(mat_vectorizer)
print("vectorizer toarray shape: "+str(mat_vectorizer.shape)+"\n after pca: "+str(mat.shape))
vocab_dict = vectorizer.vocabulary_

x_train,x_test,y_train,y_test = train_test_split(mat,lables,test_size = 0.33)

def LogReg(penalty = 'l2',c = 0.1,max_iter = 1000):
	logreg_clf = LogisticRegression(penalty = penalty, C = c, max_iter = max_iter)
	return logreg_clf

c_list = [pow(10,x) for x in range(-5,5)]
clf_list = [LogReg(c = x) for x in c_list]
fit_list = [x.fit(x_train,y_train) for x in clf_list]
train_pred_list = [x.predict(x_train) for x in clf_list]
train_acc_list = [metrics.accuracy_score(x,y_train) for x in train_pred_list]
valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in clf_list]
#print("regularizers:"+str(c_list)+"\n:training accuracies: "+str(train_acc_list)+"\nvalidation accuracies:"+str(valid_acc_list))

test_data_as_matrix = TfidfVectorizer(vocabulary = vocab_dict).fit_transform(test_text)

from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components = 300)
test_data_as_matrix = pca.fit_transform(test_data_as_matrix)
c = 1
print("test data shape after TruncatedSVD:",test_data_as_matrix.shape)
#log_pred = clf_list[5].predict_proba(test_data_as_matrix)
#sys.stdout = open("preds_test.txt","w")
#for i in range(len(log_pred)):
#	print(log_pred[i][0],",",log_pred[i][1])
#sys.stdout = sys.__stdout__
print("_________________________________________________________________________________________")
print("Regularizers:\t"+"|"+"LogReg Training Accuracies:\t|"+"\tLogReg Validation Accuracies:\t")
print("_________________________________________________________________________________________")
for i in range(len(c_list)):
	print("\t"+str(c_list[i])+"\t|"+"\t"+str(train_acc_list[i])+"\t"+"\t|"+"\t"+str(valid_acc_list[i]))
print("_________________________________________________________________________________________")



def SVM(kernel = 'rbf',c = 1):
	svc_clf = SVC(kernel = str(kernel),C = c, probability = True, verbose = False)
	return svc_clf
"""
kernels = ['rbf','poly','sigmoid','linear']
svms = {}

for i in range(len(kernels)):
	svms[kernels[i]] = [SVM(c = x, kernel = kernels[i]) for x in c_list]
"""

svm_clf_list = [SVM(c = x) for x in c_list]
svm_fit_list = [x.fit(x_train,y_train) for x in svm_clf_list]
svm_train_pred_list = [x.predict(x_train) for x in svm_clf_list]
svm_train_acc_list = [metrics.accuracy_score(x,y_train) for x in svm_train_pred_list]
svm_valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in svm_clf_list]

print("rbf kernel")
print("___________________________________________________________________________________")
print("Regularizers:\t"+"|"+"SVM Training Accuracies:\t|"+"\tSVM Validation Accuracies:\t")
print("___________________________________________________________________________________")
for i in range(len(c_list)):
	print("\t"+str(1/c_list[i])+"\t|"+"\t"+str(svm_train_acc_list[i])+"\t"+"\t|"+"\t"+str(svm_valid_acc_list[i]))
print("___________________________________________________________________________________")
"""
svm_clf_list = [SVM(c = x, kernel = 'linear') for x in c_list]
svm_fit_list = [x.fit(x_train,y_train) for x in svm_clf_list]
svm_train_pred_list = [x.predict(x_train) for x in svm_clf_list]
svm_train_acc_list = [metrics.accuracy_score(x,y_train) for x in svm_train_pred_list]
svm_valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in svm_clf_list]

print("svm linear")
print("___________________________________________________________________________________")
print("Regularizers:\t"+"|"+"SVM Training Accuracies:\t|"+"\tSVM Validation Accuracies:\t")
print("___________________________________________________________________________________")
for i in range(len(c_list)):
	print("\t"+str(1/c_list[i])+"\t|"+"\t"+str(svm_train_acc_list[i])+"\t"+"\t|"+"\t"+str(svm_valid_acc_list[i]))
print("___________________________________________________________________________________")

svm_clf_list = [SVM(c = x, kernel = 'poly') for x in c_list]
svm_fit_list = [x.fit(x_train,y_train) for x in svm_clf_list]
svm_train_pred_list = [x.predict(x_train) for x in svm_clf_list]
svm_train_acc_list = [metrics.accuracy_score(x,y_train) for x in svm_train_pred_list]
svm_valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in svm_clf_list]

print("svm poly")
print("___________________________________________________________________________________")
print("Regularizers:\t"+"|"+"SVM Training Accuracies:\t|"+"\tSVM Validation Accuracies:\t")
print("___________________________________________________________________________________")
for i in range(len(c_list)):
	print("\t"+str(1/c_list[i])+"\t|"+"\t"+str(svm_train_acc_list[i])+"\t"+"\t|"+"\t"+str(svm_valid_acc_list[i]))
print("___________________________________________________________________________________")

svm_clf_list = [SVM(c = x, kernel = 'sigmoid') for x in c_list]
svm_fit_list = [x.fit(x_train,y_train) for x in svm_clf_list]
svm_train_pred_list = [x.predict(x_train) for x in svm_clf_list]
svm_train_acc_list = [metrics.accuracy_score(x,y_train) for x in svm_train_pred_list]
svm_valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in svm_clf_list]

print("svm sigmoid")
print("___________________________________________________________________________________")
print("Regularizers:\t"+"|"+"SVM Training Accuracies:\t|"+"\tSVM Validation Accuracies:\t")
print("___________________________________________________________________________________")
for i in range(len(c_list)):
	print("\t"+str(1/c_list[i])+"\t|"+"\t"+str(svm_train_acc_list[i])+"\t"+"\t|"+"\t"+str(svm_valid_acc_list[i]))
print("___________________________________________________________________________________")


"""
log_pred = svm_clf_list[-2].predict_proba(test_data_as_matrix)
sys.stdout = open("preds_test.txt","w")
for i in range(len(log_pred)):
	print(log_pred[i][0],",",log_pred[i][1])
sys.stdout = sys.__stdout__
