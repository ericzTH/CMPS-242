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
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
#from scipy.optimize import fmin_bfgs as bfgs
from sklearn.metrics import log_loss

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
mat = vectorizer.fit_transform(text[0]).toarray()
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
print("regularizers:"+str(c_list)+"\n:training accuracies: "+str(train_acc_list)+"\nvalidation accuracies:"+str(valid_acc_list))

test_data_as_matrix = TfidfVectorizer(vocabulary = vocab_dict).fit_transform(test_text)
c = 1
log_pred = clf_list[5].predict_proba(test_data_as_matrix)
sys.stdout = open("preds_test.txt","w")
for i in range(len(log_pred)):
	print(log_pred[i][0],",",log_pred[i][1])
sys.stdout = sys.__stdout__

def SVM(kernel = 'rbf',c = 1):
	svc_clf = SVC(kernel = str(kernel),C = c, verbose = False)
	return svc_clf
kernels = ['rbf','poly','sigmoid','linear']
svms = {}
for i in range(len(kernels)):
	svms[kernels[i]] = [SVM(c = x, kernel = kernels[i]) for x in c_list]

svm_clf_list = [SVM(c = x) for x in c_list]
svm_fit_list = [x.fit(x_train,y_train) for x in svm_clf_list]
svm_train_pred_list = [x.predict(x_train) for x in svm_clf_list]
svm_train_acc_list = [metrics.accuracy_score(x,y_train) for x in svm_train_pred_list]
svm_valid_acc_list = [metrics.accuracy_score(x.predict(x_test),y_test) for x in svm_clf_list]
print("regularizers:"+str(c_list)+"\n:training accuracies: "+str(svm_train_acc_list)+"\nvalidation accuracies:"+str(svm_valid_acc_list))

c = 1
log_pred = svm_clf_list[5].predict_proba(test_data_as_matrix)
sys.stdout = open("preds_test.txt","w")
for i in range(len(log_pred)):
	print(log_pred[i][0],",",log_pred[i][1])
sys.stdout = sys.__stdout__