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

#sys.stdout = open("final.txt","a+")
#print("\n\nNew run starts here \n\n")
#print("File name: newtest1.py")
print("=======================\n\n")
os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
#print("\nlength of training data:",len(train_data))
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})
test_data = pd.read_csv("test.csv")

# Training Parameters
learning_rate = 0.01
training_steps = 15
batch_size = 1

text = train_data[0].copy()
test_text = test_data['tweet'].copy()
stop = stopwords.words('english')
"""for i in range(text.shape[0]):   
	text[i] = ' '.join([re.sub(r'[^\w\s]','',w) for w in text[i].split() if not w in stop])
for i in range(test_text.shape[0]):   
	test_text[i] = ' '.join([re.sub(r'[^\w\s]','',w) for w in test_text[i].split() if not w in stop])
"""
#tokenizing
vectorizer = TfidfVectorizer()
#temp_x = vectorizer.fit_transform(text)
train = vectorizer.fit_transform(text).toarray()
#print("text shape\n",text.shape)
#storing the vocabulary
vocab_dict = vectorizer.vocabulary_
#print("test text\n",test_text.shape)

#matrix of one hot vectors
h_iden = np.identity(len(vocab_dict))
#print("size of TfIdf vocabulary(Number of Unique Words from the data set): "
#	,len(vocab_dict),"\nTotal number of tweets: ",len(text),"\nTotal tweets in test: ",len(test_text))

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
	hidden_layer_1 = tf.Variable(tf.random_normal([num_hidden, hidden_unit_size]))#*0.01/np.sqrt(num_hidden)
	hidden_out = tf.Variable(tf.random_normal([hidden_unit_size, num_classes]))
	biases1 = tf.Variable(tf.random_normal([hidden_unit_size]))
	biases_out = tf.Variable(tf.random_normal([num_classes]))
	weights = {
		'h_l': hidden_layer_1,
		'out': hidden_out#*0.01/np.sqrt(hidden_unit_size))
	}
	
	biases = {
		'h_l': biases1,
		'out': biases_out
	}

#create the rnn

def RNN(x,weights,biases,last_word = -1):
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#x = tf.transpose(x)	
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# relu activation for ff nn hidden layer
	#print("shape of outputs",outputs.shape)
	hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l']),keep_prob = 0.7)
	# final sigmoidal output (yhat)
	yhat = tf.reshape(tf.reduce_sum(tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out']),axis = 0),[1,2]) 	
	return (yhat)
embedding_dims = 10
# graph inputs
with tf.name_scope('Embedding_matrix') :
	embedding_matrix = tf.Variable(tf.random_normal([len(vocab_dict),embedding_dims]))
with tf.name_scope('inputs'):
	#X = tf.placeholder("float", [1, None, num_input]) #(one hot vector)
	X = tf.placeholder("float", [1, None, 10]) #(one hot vector)
	Y = tf.placeholder("float", [1, num_classes])
	#last_word_in_tweet = tf.Variable(0)
with tf.name_scope('xEntropy'):
	logits = RNN(X,weights,biases)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= Y))
	prediction = tf.nn.softmax(logits)

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)		
	train_op = optimizer.minimize(loss_op)

with tf.name_scope('Accuracy'):
	#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1)), tf.float32))


#tf.summary.scalar("Accuracy",accuracy)
summary_op = tf.summary.merge([tf.summary.scalar("Clinton",prediction[0][0]),tf.summary.scalar("Trump",prediction[0][1])])

summary_op2 = tf.summary.merge([tf.summary.scalar("loss",loss_op)])
sess = tf.Session()
init = tf.global_variables_initializer()
# Start training
with sess.as_default():
	start = time.time()
	# Run the initializer
	sess.run(init) #initializing all the tf variables
	writer = tf.summary.FileWriter('M:\Course stuff\Fall 17\CMPS 242\hw5\logs', graph = tf.get_default_graph())
	#print(sess.run(tf.report_uninitialized_variables()))
	#print("timesteps: "+str(timesteps)+"\nnum_hidden in LSTM: "+str(num_hidden)+
	#	"\nFeed Fwd Neural Network hidden unit size: "+str(hidden_unit_size)+
	#	"\nLearning Rate: "+str(learning_rate)+
	#	"\nTraining Steps: "+str(training_steps)+
	#	"\nelu activation in all layers")
	for step in range(10):
	#for step in range(1,2):
		t = 0
		#for tweet in range(500):
		for tweet in range(len(train_data)):
			valid_words_indices = []
			for word in range(len(text[tweet].split(' '))):
				if text[tweet].split(' ')[word] in vocab_dict:
					valid_words_indices.append(vocab_dict[text[tweet].split(' ')[word]])
			indices_tensor = sess.run(tf.convert_to_tensor(valid_words_indices,dtype = tf.int32))
			emb_mat = tf.gather(embedding_matrix,indices_tensor, axis = 0)
			#print("emb_mat shape",emb_mat)
			m2 = sess.run(emb_mat)
			#print("run of emb_mat",m2.shape)
			matrix1 = np.reshape(m2,(1,m2.shape[0],m2.shape[1]))
			#print("\n\nembedding matrix size to be fed into LSTM cell\n",matrix1.shape)
			_,loss, acc, summary_loss = sess.run([train_op,loss_op, accuracy,summary_op2], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][tweet]).reshape(1,2)})	
			#_, pred, loss, acc, summary_pred, summary_loss = sess.run([train_op,prediction,loss_op, accuracy,summary_op,summary_op2], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][tweet]).reshape(1,2)})	
			#print("\n prediction run \n",pred)
			#writer.add_summary(summary_pred,tweet)
			writer.add_summary(summary_loss,tweet)

			if acc == 1 :
				t+=1			
			#if :
			#	pass tweet%900==0:
			#	print("For tweet "+str(tweet)+" which was said by"+str(labels_train[0][tweet])+"\nprediction of HC: "+str(pred[0][0])+" prediction of DT: "+str(pred[0][1]))	
		#print("\n matrix1 shape\n",matrix1.shape)	
		print("Training step "+str(step+1)+" acc: %f"%(t/(tweet+1))+"\n")
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


#### Testing ####
	pred_hc = []
	pred_dt = []
		#for tweet in range(5):
	for tweet in range(len(test_data)):
		valid_words_indices = []
		for word in range(len(test_text[tweet].split(' '))):
			if test_text[tweet].split(' ')[word] in vocab_dict:
				valid_words_indices.append(vocab_dict[test_text[tweet].split(' ')[word]])
		indices_tensor = sess.run(tf.convert_to_tensor(valid_words_indices,dtype = tf.int32))
		emb_mat = tf.gather(embedding_matrix,indices_tensor, axis = 0)
		#print("emb_mat shape",emb_mat)
		m2 = sess.run(emb_mat)
		#print("run of emb_mat",m2.shape)
		matrix_test = np.reshape(m2,(1,m2.shape[0],m2.shape[1]))
		temp = sess.run(prediction, feed_dict={X: matrix_test})
		pred_hc.append(temp[0][0])
		pred_dt.append(temp[0][1])
		#logits_test = sess.run([logits], feed_dict={X: matrix_test})
		#store_logits.append(logits_test) # yhats for all test tweet

	for i in range(len(pred_hc)):
		print(pred_hc[i],",",pred_dt[i])


sys.stdout = sys.__stdout__
"""
	########################### validation 4742-1743+1 = 1000 #############################
	test_len = 1000
	start = (len(train_data)-1)-1000
	#test_data = train_data[:test_len].reshape((-1, timesteps, num_input))
	test_label = labels_train[:test_len]
	t = 0
	#Validation
	for tweet in range(1000):
		#for tweet in range(len(train_data)):
		valid_words_indices = []			
		for word in range(len(text[tweet+start].split(' '))):
			if text[tweet+start].split(' ')[word] in vocab_dict:
				valid_words_indices.append(vocab_dict[text[tweet+start].split(' ')[word]])
		indices_tensor = sess.run(tf.convert_to_tensor(valid_words_indices,dtype = tf.int32))
		emb_mat = tf.gather(embedding_matrix,indices_tensor, axis = 0)
		#print("emb_mat shape",emb_mat)
		m2 = sess.run(emb_mat)
		#print("run of emb_mat",m2.shape)
		matrix1 = np.reshape(m2,(1,m2.shape[0],m2.shape[1]))
		#print("\n\nembedding matrix size to be fed into LSTM cell\n",matrix1.shape)
		_, acc= sess.run([train_op, accuracy], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][tweet+start]).reshape(1,2)})		
		if acc == 1 :
			t+=1	
	print("Validation Accuracy:", \
		t/test_len)
"""