import pandas as pd
import numpy as np
import math,os,time,itertools,sys
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import numpy as np
#sys.stdout = open('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/results.txt',"w")
global_time_init = time.time()
if os.name != 'posix':
	os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw3")
	write_path = r"M:\Course stuff\Fall 17\CMPS 242\hw3\final results"
if os.name == 'posix':
	os.chdir('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3')

#sys.stdout = open(write_path,"w")

class LogReg(object):
	def __init__(self,fname):
		self.data = pd.read_csv("new_train.csv", encoding = "ISO-8859-1")
	
	def preprocess(self):
		self.data['label'] = self.data['label'].map({'spam':1,'ham':0})
		self.y_train = self.data.iloc[:,self.data.columns=='label']
		self.text = self.data['sms']
		import nltk
		from nltk.corpus import stopwords
		stop = stopwords.words('english')
		for i in range(self.text.shape[0]):   
			self.text[i] = ' '.join([w for w in self.data['sms'][i].split() if not w in stopwords.words('english')])

		# tf-idf on train data
		from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
		vectorizer = TfidfVectorizer(stop_words = 'english')

		#temp_x = vectorizer.fit_transform(text)
		self.x_train = vectorizer.fit_transform(self.text).toarray()

		#storing the vocabulary
		vocab_dict = vectorizer.vocabulary_

		temp_x = vectorizer.fit_transform(self.text)
		self.x_train = temp_x.toarray()

		# now we have both x and y matrices which are the 
		# input text and data and corresponding spam/ham labels

		self.train = np.concatenate((self.x_train,self.y_train), axis = 1)
		#print(train.shape)

		### test data ###

		self.test_data = pd.read_csv("new_test.csv", encoding = "ISO-8859-1")
		self.test_matrix = self.test_data
		self.test_matrix['label'],self.test_matrix['sms']=self.test_data['label'].map({'spam':1,'ham':0}),self.test_data['sms']

		### VECTORIZING WITH TRAIN VOCABULARY ###
		temp_test = TfidfVectorizer(stop_words = 'english',vocabulary = vocab_dict).fit_transform(self.test_matrix['sms']).toarray()
		y_test = self.test_data.iloc[:,self.test_data.columns=='label']
		self.test = np.concatenate((temp_test,y_test), axis = 1)

		from sklearn.decomposition import PCA
		pca = PCA(n_components = 299)
		train_lowdims = pca.fit_transform(self.train[:,:-1])
		#train_lowdims.shape
		test_lowdims = pca.fit_transform(self.test[:,:-1])
		self.train2 = np.concatenate((train_lowdims,self.y_train), axis = 1)
		self.test2 = np.concatenate((test_lowdims,y_test), axis = 1)
		print(self.train2.shape,self.test2.shape)
		self.initial_w = np.random.randn(self.train.shape[1],1)*0.01/np.sqrt(self.train.shape[0])
		print("\nshape of train matrix %s\nshape of test matrix %s"%(self.train.shape,self.test.shape))
		return(self.initial_w,self.train2,self.test2)
	def sigmoid(self):
		return(1/(1+np.exp(-self)))

	def predict(self):
		for _ in range(self.shape[0]):
			if self[_,0]>=0.5:
				self[_,0] = 1
			else:
				self[_,0] = 0
		return self

	def costfn(self,reg = 0.01, penalty = 'l1'):
	m = self.train2.shape[0]
	x= self.train2[:,:-1]
	y= self.train2[:,-1]
	h = sigmoid(np.dot(x,self.initial_w))
	if penalty == 'l1':
		cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.square(np.linalg.norm(self.initial_w))
	if penalty == 'l2':
		cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.square(np.linalg.norm(self.initial_w,ord =1)) # l1 norm
	return cost[0]

	def grads(w,matrix,reg = 0.01, penalty = 'l1'):
		#print(matrix.shape)#calculates the derivative of cost function at the given w
		m = matrix.shape[0]
		x= matrix[:,:-1]
		y= matrix[:,-1]
		w_reg = w
		h = sigmoid(np.dot(x,w)) #yhat
		yhatdiffy = np.subtract(h,y.reshape(y.shape[0],1)) 
		if penalty == 'l1':
			grad = np.add((1/m)*(x.T.dot(yhatdiffy)),(reg/m)*(w)) 
		if penalty == 'l2':
			grad = np.add((1/m)*(x.T.dot(yhatdiffy)),(reg/m)*np.sign(w))
		return grad.reshape(grad.shape[0],1)


	def bgd_optimizer(w,matrix,n_iters = 100,reg = 0.01, penalty = 'l2'):

		# updates the weights matrix by computing the delta over entire input matrix #
		# w := w- sum(delta(wx-y))
		for i in range(1,n_iters+1):          
			learning_rate = 100*np.power(i,-0.9)  #eta = eta0*(iteration^-0.9)
			delta = grads(w,matrix = matrix,reg = reg, penalty = penalty)
			w = w - learning_rate * (delta) # w:= w - delta*()
			#if i%250==0:
				#print("\n\t iteration %d of %d. Cost = %r"%(i,n_iters,costfn(w,matrix = matrix ,reg = reg,penalty = penalty)))
		return w

p1 = LogReg("new_train.csv")
initial_weights, train_matrix, test_matrix = p1.preprocess()