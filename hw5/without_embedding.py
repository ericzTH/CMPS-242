'''
tokenize the train data
'''
from __future__ import division, print_function
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import tensorflow as tf
from tensorflow.contrib import rnn
import codecs,os,re,time
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
#print("\nlength of training data:",len(train_data))
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})

#tokenizing

token = []

for h in range(len(train_data)):
	token += RegexpTokenizer(r'\w+').tokenize(str(train_data[0][h]))

token=(set(token))

token1=list(token)


token2=[]
for k in token1:
	token2.append(re.sub(r'[^\w\s]','',k))

h_iden=np.identity(len(token2))


#LSTM MODEL

# creating the model
tf.reset_default_graph() 
# Training Parameters
learning_rate = 0.001
training_steps = 100 #10000
batch_size = 1 #128
display_step = 200

# Network Parameters
num_input = len(token) # MNIST data input (img shape: 28*28)
timesteps = 32 # timesteps
num_hidden = 25 # hidden layer num of features
hidden_unit_size = 10 #hidden layer for feed forward
num_classes = 2 # neural network output layer

# Define weights
weights = {
	'h_l': tf.Variable(tf.random_normal([num_hidden, hidden_unit_size])),#*0.01/np.sqrt(num_hidden)),
	'out': tf.Variable(tf.random_normal([hidden_unit_size, num_classes]))#*0.01/np.sqrt(hidden_unit_size))
}
biases = {
	'h_l': tf.Variable(tf.random_normal([hidden_unit_size])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

#create the rnn

def RNN(x,weights,biases):

# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x,axis = 0)
	#print(tf.Dimension(x))
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#print(lstm_cell)
	# Get lstm cell output
	outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)   
	#outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)   
	#print(outputs[-1],states[0][24]) #states[:][-1]
	hidden_layer = tf.nn.relu(tf.matmul(tf.reshape(states[0][-1],[1,num_hidden]), weights['h_l']) + biases['h_l']) # relu activation for ff nn hidden layer
	yhat = tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out'])# final sigmoidal output (yhat) 
	#print(yhat)# Linear activation, using rnn inner loop last output
	return (yhat)
# Initialize the variables (i.e. assign their default value)



# graph inputs

X = tf.placeholder("float", [1, timesteps, num_input]) #(one hot vector)
Y = tf.placeholder("float", [1, num_classes])
logits = RNN(X,weights,biases) 
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)		
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
	start = time.time()
	# Run the initializer
	sess.run(init) #initializing all the tf variables
	for step in range(1,training_steps+1):
		t = 0
		tf.Print(logits,[logits])
		for k in range(len(train_data)):			
			matrix1 = np.zeros(shape=(1,timesteps,len(token2)))
			for y in range(len(train_data[0][k].split(' '))):
				if (train_data[0][k]).split(' ')[y] in token2 :
					matrix1[0][:,:]=h_iden[token2.index((train_data[0][k].split(' '))[y])] # just the one hot vector 
					#matrix1 = tf.reshape(tf.cast(tf.matmul(tf.cast(matrix[0][:,:], tf.float32),embedding_tf[0][:,:]), tf.float32),[1,timesteps,embedding_dim])
			sess.run(train_op, feed_dict={X: matrix1, Y: np.asarray(labels_train[0][k]).reshape(1,2)}) # running the nn
			pred, loss, acc = sess.run([prediction,loss_op, accuracy], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][k]).reshape(1,2)})
			if acc == 1 :
				t+=1
			
			"""if k%10 == 0 and k!=0:
				print(k,acc)
				print("True label is:",labels_train[0][k])
			
				print("For tweet "+str(k)+" prediction of HC:"+str(pred[0][0])+" prediction of DT:"+str(pred[0][1]))
			"""
		if step == 1:
			print("timesteps: "+str(timesteps)+"\nnum_hidden in LSTM: "+str(num_hidden)+"\nFeed Fwd Neural Network hidden unit size: "+str(hidden_unit_size))
			print("Step"+str(step)+"\tacc  : %f"%(t/(k+1)))
		if step%10 == 0:
			print("Step"+str(step)+"\tacc  : %f"%(t/(k+1)))
			#print()
	end = time.time()
	ttl = end-start
	hrs = 0
	mins = (ttl)/60
	if mins > 60:
		hrs = mins/60
		mins %= 60
	secs = (ttl)%60
	print("Optimization Finished!")
	print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))

	# Calculate accuracy for 128 mnist test images
	#test_len = 128
	#test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
	#test_label = mnist.test.labels[:test_len]
	#print("Testing Accuracy:", \
	#	sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

