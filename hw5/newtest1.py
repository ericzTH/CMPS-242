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

#sys.stdout = open("print.txt","a+")
print("\n\nNew run starts here \n\n")
print("File name: newtest1.py")
print("=======================\n\n")
os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
#print("\nlength of training data:",len(train_data))
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})

# Training Parameters
learning_rate = 0.005
training_steps = 10
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
num_hidden = 20 # LSTM Hidden Layer size
hidden_unit_size = 8 # Feed Forward NN Hidden Layer size
num_classes = 2 # neural network output layer

# Define weights
with tf.name_scope('FFNN_Parameters'):
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
	"""
	UNSTACK REQUIRED IF WE WANT TO FEED BATCH OF TWEETS?
	pad zeros because tweets are of variable length (https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	#x = tf.unstack(x,axis = 0)
	#k = int(k)
	#x = tf.unstack(x,num = 1, axis = 1)
	#print(x,k)
	#print(tf.Dimension(x))
	"""
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# relu activation for ff nn hidden layer
	hidden_layer = tf.nn.elu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l'])
	# final sigmoidal output (yhat)
	yhat = tf.reshape(tf.reduce_sum(tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out']),axis = 0),[1,2]) 	
	return (yhat)

# graph inputs
with tf.name_scope('inputs'):
	X = tf.placeholder("float", [1, None, num_input]) #(one hot vector)
	Y = tf.placeholder("float", [1, num_classes])
	#last_word_in_tweet = tf.Variable(0)
with tf.name_scope('functions'):
	logits = RNN(X,weights,biases)
	prediction = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)		
	train_op = optimizer.minimize(loss_op)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("loss",loss_op)
tf.summary.scalar("prediction HC",prediction[0][0])
tf.summary.scalar("prediction DT",prediction[0][1])
tf.summary.scalar("Accuracy",accuracy)
sess = tf.Session()
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
# Start training
with sess.as_default():
	start = time.time()
	# Run the initializer
	sess.run(init) #initializing all the tf variables
	writer = tf.summary.FileWriter('.', graph = tf.get_default_graph())
	#print(sess.run(tf.report_uninitialized_variables()))
	print("timesteps: "+str(timesteps)+"\nnum_hidden in LSTM: "+str(num_hidden)+
		"\nFeed Fwd Neural Network hidden unit size: "+str(hidden_unit_size)+
		"\nLearning Rate: "+str(learning_rate)+
		"\nTraining Steps: "+str(training_steps)+
		"\nelu activation in all layers")
	for step in range(1,training_steps+1):
		t = 0
		tf.Print(logits,[logits])
		for k in range(len(train_data)): #len(train_data)			
		#for k in range(3000):
			matrix1 = []
			for y in range(len(text[k].split(' '))):
				if (text[k]).split(' ')[y] in vocab_dict :
					#print("Word \""+str((text[k]).split(' ')[y])+"\" found in dict") 
					matrix1.append(h_iden[vocab_dict[(text[k].split(' '))[y]]]) # append ohv of each word in the current tweet
			matrix1 = np.reshape(matrix1,(1,len(matrix1),11236))
			sess.run(train_op, feed_dict={X: matrix1, Y: np.asarray(labels_train[0][k]).reshape(1,2)}) # train the model for each tweet (not batch of tweets)
			pred, loss, acc, summary = sess.run([prediction,loss_op, accuracy,summary_op], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][k]).reshape(1,2)})	
			writer.add_summary(summary,step*1+k)
			if acc == 1 :
				t+=1			
			#if k%800==0:
				#print("For tweet "+str(k)+" which was said by"+str(labels_train[0][k])+"\nprediction of HC: "+str(pred[0][0])+" prediction of DT: "+str(pred[0][1]))	
		print("Training step "+str(step)+" acc: %f"%(t/(k+1))+"\n")
		#print("============ End of step "+str(step)+" ============\n\n\n ")
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
	# validation
	test_len = len(train_data)-3000
	#test_data = train_data[:test_len].reshape((-1, timesteps, num_input))
	test_label = labels_train[:test_len]
	t = 0			
	for k in range(test_len):
		val_matrix = []
		for y in range(len(text[3000+k].split(' '))):
			if (text[3000+k]).split(' ')[y] in vocab_dict :
				#print("Word \""+str((text[k]).split(' ')[y])+"\" found in dict") 
				val_matrix.append(h_iden[vocab_dict[(text[3000+k].split(' '))[y]]]) # append ohv of each word in the current tweet
		val_matrix = np.reshape(val_matrix,(1,len(val_matrix),len(vocab_dict)))
		sess.run(train_op, feed_dict={X: val_matrix, Y: np.asarray(labels_train[0][3000+k]).reshape(1,2)}) # train the model for each tweet (not batch of tweets)
		pred, loss, acc = sess.run([prediction,loss_op, accuracy], feed_dict={X: val_matrix,Y: np.asarray(labels_train[0][3000+k]).reshape(1,2)})	
		if acc == 1 :
			t+=1			

	print("Validation Accuracy:", \
		t/test_len)


"""
sys.stdout = sys.__stdout__