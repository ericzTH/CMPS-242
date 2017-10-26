import pandas as pd
import numpy as np
import math,os,time,itertools
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

if os.name != 'posix':
	os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw3")
	data = pd.read_csv("new_train.csv", encoding = "ISO-8859-1")
if os.name == 'posix':
	os.chdir('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3')
	data = pd.read_csv("new_train.csv", encoding = "ISO-8859-1")
#mapping spam/ham to 1/0
data['label']=data['label'].map({'spam':1,'ham':0})
y_train = data.iloc[:,data.columns=='label']
# using nltk to remove stopwords
text = data['sms']
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
for i in range(text.shape[0]):   
	text[i] = ' '.join([w for w in data['sms'][i].split() if not w in stopwords.words('english')])

# tf-idf on train data
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')

#temp_x = vectorizer.fit_transform(text)
x_train = vectorizer.fit_transform(text).toarray()

#storing the vocabulary
vocab_dict = vectorizer.vocabulary_

temp_x = vectorizer.fit_transform(text)
x_train = temp_x.toarray()

# now we have both x and y matrices which are the 
# input text and data and corresponding spam/ham labels
import numpy as np
train = np.concatenate((x_train,y_train), axis = 1)
#print(train.shape)

### test data ###

test_data = pd.read_csv("new_test.csv", encoding = "ISO-8859-1")
test_matrix = test_data
test_matrix['label'],test_matrix['sms']=test_data['label'].map({'spam':1,'ham':0}),test_data['sms']

### VECTORIZING WITH TRAIN VOCABULARY ###
temp_test = TfidfVectorizer(stop_words = 'english',vocabulary = vocab_dict).fit_transform(test_matrix['sms']).toarray()
y_test = test_data.iloc[:,test_data.columns=='label']
test = np.concatenate((temp_test,y_test), axis = 1)
print("shape of train matrix %s\nshape of test matrix %s"%(train.shape,test.shape))

def sigmoid(z):
	return(1/(1+np.exp(-z)))
#sigmoid(train[:,-1])


def predict(yhat):
	for _ in range(yhat.shape[0]):
		if yhat[_,0]>=0.5:
			yhat[_,0] = 1
		else:
			yhat[_,0] = 0
	return yhat

initial_w = np.random.randn(train.shape[1]-1,1)*0.01/np.sqrt(train.shape[0])
w = initial_w
start = time.time()
for i in range(1,101):
    w = w*(1-0.01*10*pow(i,-0.9))-10*pow(i,-0.9)*np.dot(train[:,:-1].T,(np.subtract(sigmoid(np.dot(train[:,:-1],w)),train[:,-1])))
end = time.time()
print(end-start)


w = initial_w
start = time.time()
for i in range(1,101):
    w = w*(1-0.01*10*pow(i,-0.9))-10*pow(i,-0.9)*np.dot(train[:,:-1].T,(np.subtract(1/(1+np.exp(-(np.dot(train[:,:-1],w)))),train[:,-1])))
end = time.time()
print(end-start)