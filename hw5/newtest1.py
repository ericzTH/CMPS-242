from __future__ import division, print_function
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import tensorflow as tf
from tensorflow.contrib import rnn
import codecs,os,re,time,sys
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sys.stdout = open("print.txt","w")
os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
#print("\nlength of training data:",len(train_data))
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})

# Training Parameters
learning_rate = 0.005
training_steps = 100
batch_size = 1

text = train_data[0].copy()
stop = stopwords.words('english')
for i in range(text.shape[0]):   
	text[i] = ' '.join([re.sub(r'[^\w\s]','',w) for w in text[i].split() if not w in stop])

#tokenizing
vectorizer = TfidfVectorizer(stop_words = 'english')
#temp_x = vectorizer.fit_transform(text)
train = vectorizer.fit_transform(text).toarray()
print("text shape",text.shape)
#storing the vocabulary
vocab_dict = vectorizer.vocabulary_

#matrix of one hot vectors
h_iden = np.identity(len(vocab_dict))
print("size of TfIdf vocabulary(Number of Unique Words from the data set): "
	,len(vocab_dict),"\nTotal number of tweets: ",len(text))

##LSTM MODEL

# creating the model
tf.reset_default_graph()
# Network Parameters
num_input = len(vocab_dict) # number of unique words
timesteps = 32 # timesteps
num_hidden = 25 # LSTM Hidden Layer size
hidden_unit_size = 8 # Feed Forward NN Hidden Layer size
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

def RNN(x,weights,biases,last_word = -1):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	#x = tf.unstack(x,axis = 0)
	#k = int(k)
	x = tf.unstack(x, axis = 1)
	#last_word = last_word.eval()
	#print(x,k)
	#print(tf.Dimension(x))
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#print(lstm_cell)
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	#print("states: ",len(states),"outputs: "+str(len(outputs))) #states[:][-1]
	#print("outputs Dimension",outputs[last_word].shape)
	outputs_dict = {}
	for i in range(len(outputs)):
		outputs_dict[i] = outputs[i]
	#print("\n",last_word)
	#hidden_layer = tf.nn.relu(tf.matmul(tf.reshape(outputs[last_word],[1,num_hidden]), weights['h_l']) + biases['h_l']) # relu activation for ff nn hidden layer
	#yhat = tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out'])# final sigmoidal output (yhat) 
	#print(yhat)# Linear activation, using rnn inner loop last output
	return (outputs)
# graph inputs

X = tf.placeholder("float", [1, None, num_input]) #(one hot vector)
Y = tf.placeholder("float", [1, num_classes])
last_word_in_tweet = tf.Variable(0)

#logits = RNN(X,weights,biases,last_word = last_word_in_tweet) 

def get_output(X,weights,biases,last_word):
	outputs = RNN(X,weights,biases)
	required_output = outputs[last_word]
	hidden_layer = tf.nn.relu(tf.matmul(tf.reshape(required_output,[1,num_hidden]), weights['h_l']) + biases['h_l']) # relu activation for ff nn hidden layer
	yhat = tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out'])# final sigmoidal output (yhat)
	return(yhat)
logits = get_output(X,weights,biases,last_word)

prediction = tf.nn.softmax(logits)
#convert_variable = last_word_in_tweet.eval()
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)		
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

#print("Eg:\nOriginal Tweet: "+str(train_data[0][10])+"\nAfter processing: "+str(text[10]))

sess = tf.Session()
# Start training
with sess.as_default():
	start = time.time()
	# Run the initializer
	sess.run(init) #initializing all the tf variables
	print(sess.run(tf.report_uninitialized_variables()))
	for step in range(1,training_steps+1):
		if step == 1:
			print("timesteps: "+str(timesteps)+"\nnum_hidden in LSTM: "+str(num_hidden)+"\nFeed Fwd Neural Network hidden unit size: "+str(hidden_unit_size)+"\nLearning Rate: "+str(learning_rate))
		t = 0
		tf.Print(logits,[logits])
		for k in range(len(text)): #len(train_data)			
			matrix1 = []
			for y in range(len(text[k].split(' '))):
				if (text[k]).split(' ')[y] in vocab_dict :
					#print("Word \""+str((text[k]).split(' ')[y])+"\" found in dict") 
					matrix1.append(h_iden[vocab_dict[(text[k].split(' '))[y]]]) # just the one hot vector
					#last_valid_word = y 
			print("matrix1 shape",matrix1[0].shape)
			#l_w = sess.run(convert_variable,feed_dict={last_word_in_tweet:y})
			sess.run(train_op, feed_dict={X: matrix1, Y: np.asarray(labels_train[0][k]).reshape(1,2), last_word: len(matrix1)-1}) # running the nn
			pred, loss, acc = sess.run([prediction,loss_op, accuracy], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][k]).reshape(1,2)})
		#pred, loss, acc = sess.run([prediction,loss_op, accuracy], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][k]).reshape(1,2)})	
			if acc == 1 :
				t+=1			
			if k%1000==0:
				print("For tweet "+str(k)+" prediction of HC: "+str(pred[0][0])+" prediction of DT: "+str(pred[0][1]))	
		print("Training step "+str(step)+" acc: %f"%(t/(k+1)))
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
"""
	# Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
"""