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
#get_ipython().magic('matplotlib inline')
sys.stdout = open('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/results.txt',"w")
global_time_init = time.time()
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
print("\nshape of train matrix %s\nshape of test matrix %s"%(train.shape,test.shape))

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



def costfn(w,matrix,reg = 0.01, penalty = 'l1'):
	m = matrix.shape[0]
	x= matrix[:,:-1]
	y= matrix[:,-1]
	h = sigmoid(np.dot(x,w))
	if penalty == 'l1':
		cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.square(np.linalg.norm(w))
	if penalty == 'l2':
		cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.square(np.linalg.norm(w,ord =1)) # l1 norm
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



def minibatch_optimizer(w,matrix,n_iters = 100,batch_size = 50,reg = 0.01, penalty = 'l2'):
	for i in range(1,n_iters+1):
		np.random.shuffle(matrix) #shuffle data first
		init = 0
		b_s = init+batch_size
		batches = int(matrix.shape[0]/batch_size) #total number of batches
		learning_rate = 1*np.power(i,-0.9)  #eta = eta0*(iteration^-0.9)
		if batches == matrix.shape[0]:
			for j in range(batches-1):
				#print(init,init+1)
				delta = grads(w,matrix = matrix[init:init+1,:],reg = reg, penalty = penalty)
				w = w - learning_rate * (delta)
				init += 1
			last = matrix[-2:-1,:]
			w = w - learning_rate*(grads(w,matrix = last,reg = reg, penalty = penalty))
		#if i%500==0:
			#print("\n\niteration %d of %d"%(i,n_iters))
			#current_cost = costfn(w,matrix = matrix ,reg = reg,penalty = penalty)
			#print("\ncurrent cost = ",current_cost)
		# update rule applied to each batch
		if batches != matrix.shape[0]:
			for j in range(1,batches+1):              
				if b_s > matrix.shape[0]: #in case batch size is not a multiple of shape
					b_s = b_s - matrix.shape[0]
					delta = grads(w,matrix = matrix[:-b_s,:],reg = reg, penalty = penalty)
					w = w - learning_rate * (delta)

				if b_s < matrix.shape[0]:
					delta = grads(w,matrix = matrix[init:b_s,:],reg = reg, penalty = penalty)
					w = w - learning_rate * (delta) # w:= w - delta*()
					init = b_s+1
					b_s += batch_size
	return w


# #### Helper function to plot the confusion matrix

# In[5]:


def plot_confusion_matrix(cm, classes=['ham','spam'],normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		#print("\nNormalized confusion matrix")
	else:
		#print('Confusion matrix, without normalization')
		pass

	#print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

# Implementing 10 fold cross validation to selct the $\lambda$ value. <br>
# 
# The sklearn package has a function KFold that returns cv indices based on number of splits. <br>
# 
# Iteratively called the batch gradient descent optimizer on the CV indices to build corresponding models.

# In[6]:


from sklearn.model_selection import KFold
def kfold(matrix,w,n_splits = 10,n_iters = 100,reg = 0.01, penalty = 'l2'):
	#returns training split and test split
	kf = KFold(n_splits = n_splits)
	kf.get_n_splits(matrix)
	#print(kf)
	w_list = []
	yhat_list = []
	train_acc_list = []
	validation_acc_list = []
	w_f = w
	fold = 1
	for train_index,test_index in kf.split(matrix):
		#print("\n\nfold: %d, reg: %.4f"%(fold,reg))
		#fold+=1
		#print("\nTrain indices length : %d, \t Test indices length: %d"%(len(train_index),len(test_index)))
		x_train,x_test = matrix[train_index,:],matrix[test_index,:]
		#print(x_train.shape,x_test.shape)
		w_f = bgd_optimizer(w = w,matrix = matrix[train_index,:],n_iters = n_iters,reg = reg, penalty = penalty)
		#w_list.append(w_f)
		yhat = sigmoid(np.dot(matrix[train_index,:-1],w_f))
		yhat = predict(yhat)
		train_acc_list.append(100*metrics.accuracy_score(matrix[train_index,-1],yhat))
		yhat_valid = metrics.accuracy_score(matrix[test_index,-1],predict(sigmoid(np.dot(matrix[test_index,:-1],w_f))))
		validation_acc_list.append(100*yhat_valid)
	return(np.mean(train_acc_list),np.mean(validation_acc_list),w_f)

import math
initial_w = np.random.randn(train.shape[1]-1,1)*0.001/np.sqrt(train.shape[1])
reg_list = [pow(10,x) for x in range(-3,2)]
train_acc_list = []
valid_acc_list = []
wts_fold_list = {}
start = time.time()
for reg in reg_list:
	train_acc,valid_acc,w_f = kfold(train,initial_w,n_splits = 10, n_iters =1000, reg = reg,penalty = 'l2')
	train_acc_list.append(train_acc)
	valid_acc_list.append(valid_acc)
	wts_fold_list[reg] = w_f
end = time.time()
ttl = end - start
hrs = int(ttl/3600)
ttl-=3600*hrs
mins = int(ttl/60)
sec = int((ttl)%60)
print("\nTime taken for 10fold cv with \'l2\' penalty for 1000 iterations:%d hrs %d mins %d sec :"%(hrs,mins,sec))
plt.figure()
plt.plot(np.log(reg_list),train_acc_list,label = 'training with l2')
plt.plot(np.log(reg_list),valid_acc_list,label = 'validation with l2')
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/l2 training v validation.jpg',bbox_inches='tight')
plt.title('Training , validation vs log(lambda)')

# In[28]:

j=1
plt.figure(figsize=(15,15))
plt.title('confusion matrix with l2 10fold cv')
for i in wts_fold_list.keys():
	ahat = np.dot(train[:,:-1],wts_fold_list[i])
	yhat = sigmoid(ahat)
	plt.subplot(3,3,j)
	j += 1
	yhat = predict(yhat)
	cnf_m = confusion_matrix(train[:,-1],yhat)
	accs = 100*metrics.accuracy_score(train[:,-1],yhat)
	plot_confusion_matrix(cnf_m,title = 'acc: %f reg = %f' %(accs,i))
	print("\n\'l2\' accuracy with reg = %.5f : %.4f"%(i,accs))
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/10fold CV results with l2.jpg',bbox_inches='tight')

# In[26]:


train_acc_list_l1 = []
valid_acc_list_l1 = []
wts_fold_list_l1 = {}
start = time.time()
for reg in reg_list:
	train_acc,valid_acc,w_f = kfold(train,initial_w,n_splits = 10, n_iters =1000, reg = reg,penalty = 'l1')
	train_acc_list_l1.append(train_acc)
	valid_acc_list_l1.append(valid_acc)
	wts_fold_list_l1[reg] = w_f
end = time.time()
ttl = end - start
hrs = int(ttl/3600)
ttl-=3600*hrs
mins = int(ttl/60)
sec = int((ttl)%60)
print("\nTime taken for 10fold cv with \'l1\' penalty for 1000 iterations:%d hrs %d mins %d sec :"%(hrs,mins,sec))


# In[27]:

plt.figure()
plt.plot(np.log(reg_list),train_acc_list_l1,label = 'training')
plt.plot(np.log(reg_list),valid_acc_list_l1,label = 'validation')
plt.legend()
plt.title('Train vs Validation with \'l2\' and \'l1\' penalty')
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/l1 training v validation.jpg',bbox_inches='tight')

plt.figure(figsize=(15,15))
j = 1
for i in wts_fold_list_l1.keys():
	ahat = np.dot(train[:,:-1],wts_fold_list_l1[i])
	yhat = sigmoid(ahat)
	plt.subplot(3,3,j)
	j+=1
	yhat = predict(yhat)
	cnf_m = confusion_matrix(train[:,-1],yhat)
	accs = 100*metrics.accuracy_score(train[:,-1],yhat)
	#plt.figure(figsize = (20,20))
	plot_confusion_matrix(cnf_m,title = 'using \'l1\' penalty acc: %f reg = %f' %(accs,i))
	##plt.show()
	print("\n\'l1\'.accuracy with reg = %.5f : %.4f"%(i,accs))
plt.title('confusion matrix with l1 10fold cv')
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/10fold CV results with l1.jpg',bbox_inches='tight')

# From the confusion matrix plots we can see that using $\lambda$ = 0.001 gives the optimal result. <br>
# 
# In the cell below, let's using the logistic regression hypothesis $$ y^{'} = \big(\,log\,(sigmoid(X.w))\big)$$ on the test data and find the accuracy, confusion matrix. This is the final result.

# In[9]:

plt.figure(figsize = (15,15))
plt.title('confusion matrix with l2 on test data')
w_final = wts_fold_list[0.001]
yhat = predict(sigmoid(np.dot(test[:,:-1],w_final)))
print("\nAccuracy on test data with lambda : %f = %f"%(0.001,100*(metrics.accuracy_score(test[:,-1],yhat))))
cnfm_t = confusion_matrix(test[:,-1],yhat)
plot_confusion_matrix(cnfm_t,title = 'Result on test data with \'l2\' penalty, reg = %f,acc: %f' %(0.001,100*(metrics.accuracy_score(test[:,-1],yhat))))
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/Result on test data with l2.jpg',bbox_inches='tight')

# ## Extra Cedit
# Building a new model with $\lambda$ obtained above but with a different penalty term using the l1 norm instead of the l2 norm.

# In[10]:

plt.figure(figsize = (15,15))
plt.title('confusion matrix with l1 on test data')
print(train.shape)
start = time.time()
w_withl1 = bgd_optimizer(matrix = train,w = initial_w,n_iters = 500, reg = 0.001,penalty = 'l1')
end = time.time()
print("\nTime taken for batch gradient descent = %f"%(end-start))
yhat_withl1 = predict(sigmoid(np.dot(test[:,:-1],w_withl1)))
print("\nAccuracy on test data with lambda : %f = %f"%(0.001,100*(metrics.accuracy_score(test[:,-1],yhat_withl1))))
cnfm_t = confusion_matrix(test[:,-1],yhat_withl1)
plot_confusion_matrix(cnfm_t,title = 'Result on test data with \'l1\' penalty, reg = %f,acc: %f' %(0.001,100*(metrics.accuracy_score(test[:,-1],yhat_withl1))))
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/Result on test data with l1.jpg',bbox_inches='tight')
#plt.show()

# Models built with same hyperparameters but different penalties gave us different accuracy on the test set. <br> With l2 penalty model performing better than the model built using l1 penalty.

# ** Building the model with same hyperparameters but using mini batch gradient descent with batch size = 50 **
# 
# *test accuracy and confusion matrix shown *

# In[16]:

plt.figure(figsize = (10,10))
plt.title('confusion matrix mini bgd and batch size 50, on test data')
initial_w = np.random.randn(train.shape[1]-1,1)
start = time.time()
w = minibatch_optimizer(w = initial_w, matrix = train,n_iters = 500, batch_size = 50, reg = 0.001)
end = time.time()
print("\nTime taken for mini batch gradient descent with batch size = 50 :%f"%(end-start))
yhat_mb = sigmoid(np.dot(test[:,:-1],w))
yhat_mb = predict(yhat_mb)
print("\naccuracy:%.4f"%(metrics.accuracy_score(test[:,-1],yhat_mb))) # always ytrue and ypred
plot_confusion_matrix(cnfm_t,title = 'Result on test data with \'l2\' penalty using mini batch GD, reg = %f,acc: %f' %(0.001,100*(metrics.accuracy_score(test[:,-1],yhat_mb))))
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/Result on test data with l2 penalty using MBGD.jpg',bbox_inches='tight')
#plt.show()

# ** Building the model with same hyperparameters but using stochastic gradient descent**
# 
# *test accuracy and confusion matrix shown *

# In[24]:

plt.figure(figsize = (10,10))
plt.title('confusion matrix SGD, on test data')
initial_w = np.random.randn(train.shape[1]-1,1)
start = time.time()
w = minibatch_optimizer(w = w, matrix = train,n_iters =500, batch_size = 1, reg = 0.001)
end = time.time()
print("\nTime taken for stochastic gradient descent:%f"%(end-start))
yhat_sgd = sigmoid(np.dot(test[:,:-1],w))
yhat_sgd = predict(yhat_mb)
print("\naccuracy:%.4f"%(metrics.accuracy_score(test[:,-1],yhat_sgd))) # always ytrue and ypred
plot_confusion_matrix(cnfm_t,title = 'Result on test data with \'l2\' penalty using SGD, reg = %f,acc: %f' %(0.001,100*(metrics.accuracy_score(test[:,-1],yhat_sgd))))
plt.savefig('/media/sanjay/New Volume/Course stuff/Fall 17/CMPS 242/hw3/final results/Result on test data with l2 penalty using SGD.jpg',bbox_inches='tight')


# ##### Inlcuding bias term and then not regularizing it:
# 
# *Concatenating ones vector on the first column*
# 
# Note that the last column is still the labels column.

train_bias = np.c_[np.ones((train.shape[0],1)),train]
train_bias.shape


train_bias_x = train_bias[:,:-1]
y = train_bias[:,-1]
w_bias = np.random.randn(train_bias_x.shape[1])*0.01/np.sqrt(train_bias_x.shape[0])
def cost_bias(initial_w,x,y,reg):
	m = y.size
	h = sigmoid(np.dot(x,initial_w))
	cost_reg = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(w[1:,:]))
	return cost_reg
#print(cost_bias(w_bias,train_bias_x,y,0.01))

def grads_bias(w,x,y,reg = 0.01, penalty = 'l2'):
	w_reg = w.reshape((w.shape[0],1))#calculates the derivative of cost function at the given w
	m = y.size
	#w_reg = w
	w_reg[:,0]=0 #grad = np.add((1/m)*(x.T.dot(yhatdiffy)),(reg/m)*(w)) 
	h = sigmoid(np.dot(x,w_reg)) #yhat
	yhatdiffy = np.subtract(h,y.reshape(y.shape[0],1)) 
	if penalty == 'l2':
		grad = np.add((1/m)*(x.T.dot(yhatdiffy)),(reg/m)*(w_reg)) 
	return grad.reshape(grad.shape[0],1)
#print(grads_bias(w_bias,train_bias_x,y,0.01).shape)


def bgd_optimizer_bias(w,x,y,n_iters = 500,reg = 0.01, penalty = 'l2'):

	# updates the weights matrix by computing the delta over entire input matrix #
	# w := w- sum(delta(wx-y))
	for i in range(1,n_iters+1):   
		learning_rate = 5*np.power(i,-0.9)  #eta = eta0*(iteration^-0.9)
		delta = grads_bias(w,x,y,reg = reg, penalty = penalty)
		w = (w.reshape((w.shape[0],1))-learning_rate * (delta)) # w:= w - delta*()
		if i%250==0:
			print("\n\t iteration %d of %d. Cost = %r"%(i,n_iters,cost_bias(w,x,y,reg = reg)[0]))
	return w
w_opt = bgd_optimizer_bias(w_bias,train_bias_x,y,reg = 0.001,n_iters = 500)
yhat_bias = predict(sigmoid(np.dot(train_bias_x,w_opt).reshape((train_bias_x.shape[0],1))))
print("\nAccuracy with bias and same model %f"%(100*metrics.accuracy_score(y,yhat_bias)))


# In[31]:


print("\nTrain accuracy with l2",train_acc_list,"\nValidation accuracy with l2",valid_acc_list,"\nTrain accuracy with l1",train_acc_list_l1,"\nValidation accuracy with l1",valid_acc_list_l1)


# In[33]:


start = time.time()
w = minibatch_optimizer(w = w, matrix = train,n_iters =500, batch_size = 3000, reg = 0.001)
end = time.time()
print('mini batch with batch size = 3000 and n_iters = 500',end-start)


# In[34]:


start = time.time()
w = bgd_optimizer(w = w, matrix = train,n_iters = 500,reg = 0.001)
end = time.time()
print('BGD time n_iters = 500:',end-start)
print("\nNote: all n_iters are set to 1000 no matter what the above lines say.")
global_time_finish = time.time()
global_ttl = global_time_finish - global_time_init
hrs = int(global_ttl/3600)
global_ttl-=3600*hrs
mins = int(global_ttl/60)
sec = int((global_ttl)%60)
print("\nTotal time taken :%d hrs %d mins %d sec :"%(hrs,mins,sec))
sys.stdout = sys.__stdout__

# In[ ]:


# coding: utf-8

# # CMPS 242 Homework Assignment 3
# ## Sanjay Krishna Gouda
# <br><br>
# 
# #### Results
# 
# | $\lambda$                            | 0.001   | 0.01  | 0.1    | 1     | 10    |
# | -------------                        |:-------:| :----:| :----: |:-----:|:-----:|
# | Training accuracy (with l2 penalty)  | 97.44   | 97.41 | 97.27  | 95.23 | 81.13 |
# | validation accuracy (with l2 penalty)| **95.67**   | 95.53 | 95.46  | 93.30 | 79.69  |
# | Test accuracy (with l2 penalty)      | **94.13**   |   -   |   -    |   -   |   -   |
# | Training accuracy (with l1 penalty)  | 97.44   | 97.44 | 97.43  | 95.41 | 97.07 |
# | validation accuracy (with l1 penalty)| **95.76**   | 95.76 | 95.76  | 95.66 | 95.23  |
# | Test accuracy (with l1 penalty)      | **94.13**   |   -   |   -    |   -   |   -   |
# <br>
# 
# 
# | $\lambda$                            | True Negatives   | True Positives | False Positives| False Negatives|
# | -------------                        |:-------:|:------:| :----:| :----: |:-----:|:------:|:-----:|:-----: |
# |0.001                                 | 2556(l2) 2556(l1)|366(l2)  366(l1)|27(l2) 27(l1)   | 51(l2) 51(l2)  |
# |0.01                                  | 2555(l2) 2556(l1)|365(l2)  366(l1)|28(l2) 27(l1)   | 52(l2) 51(l2)  |
# |0.1                                   | 2560(l2) 2555(l1)|357(l2)  366(l1)|27(l2) 28(l1)   | 60(l2) 51(l2)  |
# |1                                     | 2515(l2) 2555(l1)|333(l2)  365(l1)|68(l2) 28(l1)   | 84(l2) 52(l2)  |
# |10                                    | 2193(l2) 2557(l1)|250(l2)  350(l1)|390(l2) 26(l1)  | 167(l2) 67(l2) |
# 
# *Time taken for stochastic dradient descent :40.63 seconds* <br>
# *Time taken for mini batch dradient descent with batch size = 50 :15.85 seconds *<br>
# *Time taken for batch gradient descent = 218 seconds*<br>
# ### Implementing Logistic Regression with Batch Gradient Descent. <br>
# #### Logistic Regression hypothesis:
# $$ y^{'} = sigmoid(X.w) $$
# 
# With cost function 
# $$ J(w) = \frac{1}{m}\sum_{i=1}^{m}\big[-y^{(i)}\, log\,( h_w\,(x^{(i)}))-(1-y^{(i)})\,log\,(1-h_w(x^{(i)}))\big]+\frac{\lambda}{2m}\sum_{j=1}^{m}w_j^2$$
# or in the vectorized form 
# $$ J(w) = \frac{1}{m}\big((\,log\,(g(Xw))^Ty+(\,log\,(1-g(Xw))^T(1-y)\big)+\frac{\lambda}{m}(\vert\vert w \vert\vert^2_2)$$
# where m is the number of examples and g(z) is the sigmoidal activation given by $$ g(z)=\frac{1}{1+e^{−z}} $$
# 
# Choosing regularizer (the $\lambda$) value based on 10 fold cross validation scores. <br>
# 
# Adding regularizer to the cost function. Initially, just the L2 norm of weights but in later cells, other norms. (for extra credit). <br>
# 
# In the cell below: <br>
# * import train and test csvs
# * map spam/ham to 1/0
# * remove stop words from train file
# * use tf-idf and vectorize the train file
# * use the vocabulary of the above vectorization and vectorize the test file
# * print the shapes of train and test matrices with last column being the mapped labels<br>
# 
# *the matrix built using tfidfvectorizer normalizes the matrix by default using norm = 'l2'*


# ** bgd_optimizer(w,matrix,n_iters = 100,reg = 0.01, penalty = 'l2'): ** <br>
# ##### Batch gradient descent function
# Inputs:
# * n_iters = Number of times the weights update process is repeated.
# * reg = the $\lambda$ (regularizer) value. Defaults to 0.01
# * penalty = if 'l1', uses l1 norm regularizer and if 'l2' uses l2 norm regularizer. 
# * w = initial weights matrix
# * matrix = the matrix on which to train on.
# Returns:
# * w_opt = Returns the weight vector after finishing the update mechanism/ optmization.
# 
# **Algorithm:**<br>
# * Actual Learning Rate:
# $$\eta = \eta_0.t^{-\alpha} $$
# where $\alpha =0.9$
# here I set $\eta_0 = 100$ and used less number of iterations so that it does not take too long.
# (not submitted: tried different iterations with different etas but got similar results.)
# * Update Rule:
# $$ w:= w-\eta.grads(weights = w, matrix = entire input matrix, reg = reg) $$
# * Updates the parameters just once after computing the grads on entire matrix.

# In[3]:



# ### Model Selection
# Below cell gives the results of 10fold cv with 1000 iterations using different $\lambda$.
# 
# For spam detection, misclassification of good texts as spam is more punishable. <br>
# 
# Hence, along with accuracy, we may consider selecting a model that has the least number of misclassifications of good texts as spam and highest number of correct spam detection.<br>
# 
# i.e the model should have better true positive and false negative scores along with accuracy.<br>
# 
# Plotting the confusion matrix for the models obtained with different $\lambda$s gives a better visualization of the scenario.<br>
# 
# Building the accuracy vs $ log(\lambda) $ plot and the confusion matrix plots for all the 10 fold cv models with different regularizers. <br>
# 
# *Here CV method was applied using the l2 norm only.* 

# In[25]:

# **costfn** <br>
# Inputs : weights vector, training matrix (including labels) and regularizer that defaults to $\lambda$ = 0.01 <br>
# Splits the input training matrix into x and y matrices where y is the last column and x is all columns except the last. This matrix *x* is the one I use for training. <br>
# Returns the cost based on the equation:
# $$ J(w) = \frac{1}{m}\big((\,log\,(g(Xw))^Ty+(\,log\,(1-g(Xw))^T(1-y)\big)$$ without regularization and 
# $$ J(w) = \frac{1}{m}\big((\,log\,(g(Xw))^Ty+(\,log\,(1-g(Xw))^T(1-y)\big)+\frac{\lambda}{m}(\vert\vert w \vert\vert^2_2)$$
# with L2 regularization. Where m is the number of examples and g(z) is the sigmoidal activation given by $$ g(z)=\frac{1}{1+e^{−z}} $$
# 
# ** grads ** <br>
# Inputs: 
# Returns the gradient of cost function taken with respect to w. This is given by
# $$ \frac{\delta J(w)}{\delta w_{j}} = \frac{1}{m} X^T(g(Xw)-y)$$ without regularization and
# $$ \frac{\delta J(w)}{\delta w_{j}} = \frac{1}{m} X^T(g(Xw)-y) + \frac{\lambda}{m}w_{j}$$ with L2 regularization

# In[2]:

# ** minibatch_optimizer(n_iters,batch_size,w,matrix,reg = 0.01, print_cost = True): ** <br>
# Inputs:
# * n_iters = Number of times the weights update process is repeated.
# * batch_size = Number of examples using which the costfn and grads functions are used to update the weights vector.
# * reg = the $\lambda$ (regularizer) value. Defaults to 0.01
# * penalty = if 'l1', uses l1 norm regularizer and if 'l2' uses l2 norm regularizer. 
# * w = initial weights matrix
# * matrix = the matrix on which to train on.
# Returns:
# * w_opt = Returns the weight vector after finishing the update mechanism/ optmization.
# 
# **Algorithm:**<br>
# * Actual Learning Rate:
# $$\eta = \eta_0.t^{-\alpha} $$
# where $\alpha =0.9$
# here I set $\eta_0 = 1$
# * Update Rule:
# $$ w:= w-\eta.grads(weights = w, matrix = current_batch, reg = reg) $$
# * Updates are applied iteratively using the batches.
# * Batch construction: <br>
# Used two pointers with fixed distance which equals the batch size and iteratively around the values each time passing these two pointers as indices of the matrix in the grads() function call.
# 
# ** Note ** : Setting batch_size to 1 gives Stochastic Gradient Descent and setting batch_size to number of examples gives batch gradient descent.

# In[4]:


# To not regularize the bias term the vectorized equations are as follows:
# Cost - 
# $$ J(w) = \frac{1}{m}\big((\,log\,(g(Xw))^Ty+(\,log\,(1-g(Xw))^T(1-y)\big)+\frac{\lambda}{m}(\vert\vert w[1:,] \vert\vert^2_2)$$
# 
# grads -
# $w[0,] = 0$ Because the first column is always 1 a constant, it's derivative would be 0. So setting that column values to 0
# $$ \frac{\delta J(w)}{\delta w_{j}} = \frac{1}{m} X^T(g(Xw)-y) + \frac{\lambda}{m}w$$ with L2 regularization

# In[19]:

