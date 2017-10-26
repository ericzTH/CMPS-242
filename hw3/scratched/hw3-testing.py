
# coding: utf-8

# In[1]:


import pandas as pd
import os
os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw3")
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
print(train.shape)

### test data ###

test_data = pd.read_csv("new_test.csv", encoding = "ISO-8859-1")
test_matrix = test_data
test_matrix['label'],test_matrix['sms']=test_data['label'].map({'spam':1,'ham':0}),test_data['sms']
### VECTORIZING WITH TRAIN VOCABULARY ###
temp_test = TfidfVectorizer(stop_words = 'english',vocabulary = vocab_dict).fit_transform(test_matrix['sms']).toarray()
y_test = test_data.iloc[:,test_data.columns=='label']
test = np.concatenate((temp_test,y_test), axis = 1)


# In[2]:


print("shape of train matrix %s\nshape of test matrix %s"%(train.shape,test.shape))


# In[32]:


## logistic regression ##
def sigmoid(z):
    return(1/(1+np.exp(-z)))
#sigmoid(train[:,-1])

def costfn(w,matrix,reg = 0.01):
    m = matrix.shape[0]
    x= matrix[:,:-1]
    y= matrix[:,-1]
    h = sigmoid(np.dot(x,w))
    #cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.sum(np.square(w[2:,:]))
    cost = -(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2*m))*np.sum(np.square(w[:,:]))
    return cost[0]

def grads(w,matrix,reg = 0.01):
    m = matrix.shape[0]
    x= matrix[:,:-1]
    y= matrix[:,-1]
    w_reg = w
    w_reg[0,:] = 0     #dc/dw
    h = sigmoid(np.dot(x,w)) #yhat
    #print(h.shape,y.shape)
    yhatdiffy = np.subtract(h,y.reshape(y.shape[0],1)) 
    grad = np.add((1/m)*(x.T.dot(yhatdiffy)),(reg/m)*(w) )  #regularized w->w_reg
    #print(grad.shape)
    return grad.reshape(grad.shape[0],1)

#### BATCH GRADIENT DESCENT ####

def bgd_optimizer(n_iters,batch_size,learning_rate,w,matrix,reg = 0.01, print_cost = True):
	m = matrix.shape[0]
	x= matrix[:,:-1]
	y= matrix[:,-1]
	 
	grad_list = []
	if batch_size > m:
		return None
	while n_iters>0:
		alpha = 10*np.power(n_iters,-learning_rate) 		 # this is the learning rate with eta = 10
		init,b_s = 0,batch_size
		counter = 0#int(batch_size / m)						 # alpha = eta * (iteration)^-learnign_rate
		if print_cost == True and n_iters%100 == 0:
			print("n_iter = %d, cost = %.16f, learning rate = %.8f"%(n_iters,costfn(w,matrix,reg = reg),alpha))
		print("current iteration:",n_iters)
		while counter < ( m / batch_size ): 
			#print(counter+1)
			#print(init,b_s)
			#print("n_iter = ",n_iters,"batch = ",counter+1,"matrix shape",matrix[init:b_s,:].shape)
			delta = grads(w,matrix[init:b_s,:],reg=reg) # delta = d/dw(cost)
			grad_list.append(delta)
			#print(init,b_s)
			w = np.subtract(w , (alpha/batch_size)*(delta))  # w = w - alpha*delta. or w - (alpha/batch_size)*delta
			init = b_s
			b_s += batch_size
			counter += 1
		delta_last = grads(w,matrix[:-batch_size,:],reg = reg)
		w = np.subtract(w, (alpha/batch_size)*delta_last)
		grad_list.append(delta_last)
		#print(len(grad_list))	
		n_iters -= 1
	
	return(w,grad_list)

### ACCURACY HELPER ###
from sklearn import metrics

def acc(yhat,y):
	for _ in range(yhat.shape[0]):
		if yhat[_,0]>=0.5:
			yhat[_,0] = 1 
		yhat[_,0] = 0
	return(metrics.accuracy_score(yhat,y))


### kfoldS ###

def kFolds(n_folds,matrix,reg = 0.01,n_iters = 1000,batch_size = 50,learning_rate = 0.9):
	#start = time.time()
	np.random.shuffle(matrix) # randomly shuffling the data set just once, 
							  # need not use this because of below line.
	indices = np.random.permutation(matrix.shape[0])
	chunk_size = matrix.shape[0]/n_folds
	yhat_list = []
	accs_list = []
	yhat_train_list = []
	accs_train_list = []
	indices_dict = {}
	c = 0
	initial_w = np.random.randn(matrix.shape[1]-1,1)*0.01/np.sqrt(matrix.shape[0]) #random weight initialization
	for i in range(n_folds-1):
		#print(c,c+chunk_size)
		print("fold: ",i+1,"reg:",reg)
		indices_dict[i+1] = indices[c:c+int(chunk_size)]
		c += int(chunk_size)
	
		list1 = [x+1 for x in range(matrix.shape[0])]
		list2 = [x-1 for x in list1 if x not in indices_dict[i+1]]
		#print(len(list2))
		temp_train = matrix[list2,:]
		if temp_train.shape[0]>int(chunk_size*(n_folds-1)):
			temp_train=temp_train[1:,:]
		#print("i = ",i,"shape of train = ",temp_train[:,:-1].shape,"shape of test = ",matrix[indices_dict[i+1],:-1].shape)
		
		w_opt,deltas = bgd_optimizer(n_iters = n_iters,            			# number of iterations
									batch_size = batch_size,            	# batch size
									learning_rate = learning_rate,        	# alpha
									w = initial_w,              			# initial weights
									matrix = temp_train[:,:],   			# training chunk of training set
									print_cost = False)
		yhat_train_list.append(sigmoid(np.dot(temp_train[:,:-1],w_opt))) 	# for the training accuracies
		accs_train_list.append(acc(yhat_train_list[-1],temp_train[:,-1]))
		yhat_list.append(sigmoid(np.dot(matrix[indices_dict[i+1],:-1],w_opt))) # yhat = sigmoid(Xtest.w*)
		accs_list.append(acc(yhat_list[-1],matrix[indices_dict[i+1],-1]))       # returns accuracy
	# last fold here
	last_list = list1[:-int(chunk_size)]
	for x in range(len(last_list)):
		last_list[x] = -last_list[x]
	print("last fold")
	w_opt_last,deltas = bgd_optimizer(n_iters = n_iters,            
									batch_size = batch_size,            
									learning_rate = learning_rate,        
									w = initial_w,              
									matrix = matrix[last_list,:],   
									print_cost = False)	
	yhat_train_list.append(sigmoid(np.dot(temp_train[:,:-1],w_opt_last))) 	# for the training accuracies
	accs_train_list.append(acc(yhat_train_list[-1],temp_train[:,-1]))
	yhat_list.append(sigmoid(np.dot(matrix[indices_dict[i+1],:-1],w_opt_last))) # yhat = sigmoid(Xtest.w*)
	accs_list.append(acc(yhat_list[-1],matrix[indices_dict[i+1],-1]))       # returns accuracy
	#end = time.time()
	#print("\ntime taken for 10Fold CV: %.4f seconds"%(end-start))
	return(np.mean(accs_train_list)*100,np.mean(accs_list)*100) # accuracy averaged over n_folds


# In[33]:


### CELL FOR FUNCTION CALLS AND PLOTTING ### #and also debugging :p#
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
#get_ipython().magic('matplotlib inline')
acc_train_kfold = []
acc_test_kfold = []
reg_list = [ 0.001,0.01, 0.1, 1, 5, 10]
for reg in reg_list:
	a,b = kFolds(10,train,batch_size = 1000,n_iters = 1,learning_rate = 0.9, reg = reg)
	acc_train_kfold.append(a)
	acc_test_kfold.append(b)
plt.plot((reg_list),acc_train_kfold,label = 'Training')
plt.plot((reg_list),acc_test_kfold,label = 'Validation')
plt.legend()


# In[34]:


reg_list,acc_train_kfold,acc_test_kfold


# In[ ]:


acc_train_kfold = []
acc_test_kfold = []
import math
reg_list = [pow(10,x) for x in range(-5,5) ]
for reg in reg_list:
	a,b = kFolds(10,train,batch_size = 100,n_iters = 500,learning_rate = 0.9, reg = reg)
	acc_train_kfold.append(a)
	acc_test_kfold.append(b)
plt.plot(np.log(reg_list),acc_train_kfold,label = 'Training')
plt.plot(np.log(reg_list),acc_test_kfold,label = 'Validation')
plt.legend()


# In[ ]:




