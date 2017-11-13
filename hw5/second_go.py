from __future__ import division, print_function
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
import tensorflow as tf
from tensorflow.contrib import rnn
import codecs,os,re,time,sys
import numpy as np
import pandas as pd


if os.name != 'posix':
	os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
#print("\nlength of training data:",len(train_data))
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})
test_data = pd.read_csv("test.csv")

# Training Parameters
learning_rate = 0.001
training_steps = 300
batch_size = 1

text = train_data[0].copy()
test_text = test_data['tweet'].copy()
stop = stopwords.words('english')
"""for i in range(text.shape[0]):   
	text[i] = ' '.join([re.sub(r'[^\w\s]','',w) for w in text[i].split() if not w in stop])
for i in range(test_text.shape[0]):   
	test_text[i] = ' '.join([re.sub(r'[^\w\s]','',w) for w in test_text[i].split() if not w in stop])
"""

def create_token_lexicon(trains):
	print("Creating lexicon.")
	tknzr = TweetTokenizer()
	lexicon = set()
	for i in range(len(trains)):
		lexicon.update(tknzr.tokenize(trains[i]))
	w_counts = Counter(lexicon)
	print("Lexicon has ", len(lexicon), " entries")
	#print(w_counts)
	return list(lexicon)
token_list = create_token_lexicon(text)
vocab_dict = {}
for i in range(len(token_list)):
	if token_list[i] not in vocab_dict:
		vocab_dict[token_list[i]] = i
vocab_dict['today.'] = vocab_dict['today']
#print("test text\n",test_text.shape)

"""
vocab_dict['#DebateNight']=vocab_dict['debatenight']
vocab_dict['#debatenight']=vocab_dict['debatenight']
vocab_dict['The'] = vocab_dict['the']
 
vocab_dict['Trump'] = vocab_dict['trump']
vocab_dict['Hillary'] = vocab_dict['hillary']
vocab_dict['HillaryClinton'] = vocab_dict['hillaryclinton']
vocab_dict['Nuclear'] = vocab_dict['nuclear']
vocab_dict['action.'] = vocab_dict['action']
vocab_dict['BAD'] = vocab_dict['bad']
vocab_dict['returns.'] = vocab_dict['returns']
vocab_dict['No'] = vocab_dict['no']
#matrix of one hot vectors
"""

h_iden = np.identity(len(vocab_dict))
print("size of TfIdf vocabulary(Number of Unique Words from the data set): ",len(vocab_dict),
	"\nTotal number of tweets: ",len(text),"\nTotal tweets in test: ",len(test_text))

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
with tf.name_scope('rnn'):
	hidden_layer_1 = tf.Variable(tf.random_normal([num_hidden, hidden_unit_size]))#*0.01/np.sqrt(num_hidden)
	hidden_out = tf.Variable(tf.random_normal([hidden_unit_size, num_classes]))#*0.01/np.sqrt(hidden_unit_size)
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
"""
with tf.name_scope('embedding_matrix') :
	embeddings = tf.get_variable('embedding_matrix', [len(vocab_dict), embedding_dims])
	#embedding_matrix = tf.Variable(tf.random_normal([len(vocab_dict),embedding_dims]),trainable = True)#*0.01
def RNN(x,weights,biases,embedding_matrix):
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#x = tf.transpose(x)	
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# relu activation for ff nn hidden layer
	#print("shape of outputs",outputs.shape)
	hidden_layer = tf.nn.dropout(tf.nn.elu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l']),keep_prob = 0.7)
	# final sigmoidal output (yhat)
	yhat = tf.reshape(tf.reduce_sum(tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out']),axis = 0),[1,2])
	return (yhat)
"""
def RNN(indices,weights,biases):
	#print(tf.shape(indices))
	embedding_mat = tf.get_variable('embedding_matrix',[len(vocab_dict),embedding_dims])
	x = tf.nn.embedding_lookup(embedding_mat,indices) #,[1,tf.shape(indices),embedding_dims])
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#x = tf.transpose(x)	
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# relu activation for ff nn hidden layer
	print("shape of outputs",outputs.shape)
	hidden_layer = tf.nn.dropout(tf.nn.elu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l']),keep_prob = 0.7)
	# final sigmoidal output (yhat)
	yhat = tf.reshape(tf.reduce_sum(tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out']),axis = 0),[1,2])
	return (yhat)	
embedding_dims = 10
# graph inputs

with tf.name_scope('inputs'):
	indices = tf.placeholder(tf.int32,shape=[None,1])
	#len_indices = tf.placeholder()
	#X = tf.placeholder("float", [1, None, num_input]) #(one hot vector)
	X = tf.placeholder("float", [1, None, embedding_dims]) #(one hot vector)
	Y = tf.placeholder("float", [1, num_classes])
	#last_word_in_tweet = tf.Variable(0)
with tf.name_scope('xEntropy'):
	logits = RNN(indices,weights,biases)
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
	writer = tf.summary.FileWriter('\logs', graph = tf.get_default_graph())
	#for step in range(training_steps):
	for step in range(20):
		t = 0
		step_start = time.time()
		#for tweet in range(10):
		for tweet in range(len(train_data)):
			#print(tweet)
			valid_words_indices = []
			for word in range(len(text[tweet].split(' '))):
				if text[tweet].split(' ')[word] in vocab_dict:
					valid_words_indices.append(vocab_dict[text[tweet].split(' ')[word]])
			valid_words_indices = np.asarray(valid_words_indices).reshape(len(valid_words_indices),1)
			#if len(valid_words_indices)!=0:
			_,loss, acc, summary_loss = sess.run([train_op,loss_op, accuracy,summary_op2], feed_dict={indices: valid_words_indices,Y: np.asarray(labels_train[0][tweet]).reshape(1,2)})	
			#_, pred, loss, acc, summary_pred, summary_loss = sess.run([train_op,prediction,loss_op, accuracy,summary_op,summary_op2], feed_dict={X: matrix1,Y: np.asarray(labels_train[0][tweet]).reshape(1,2)})	
			#print("\n prediction run \n",pred)
			#writer.add_summary(summary_pred,tweet)

			if acc == 1 :
				t+=1			
			#if :
			#	pass tweet%900==0:
			#	print("For tweet "+str(tweet)+" which was said by"+str(labels_train[0][tweet])+"\nprediction of HC: "+str(pred[0][0])+" prediction of DT: "+str(pred[0][1]))	
		#print("\n matrix1 shape\n",matrix1.shape)	
		writer.add_summary(summary_loss,tweet)
		step_end = time.time()
		step_ttl = step_end-step_start
		step_hrs = 0
		step_mins = (step_ttl)/60
		if step_mins > 60:
			step_hrs = mins/60
			step_mins %= 60
		step_secs = (step_ttl)%60
		print("Training step "+str(step+1)+" acc: %f"%(t/(tweet+1))+"\tLoss: "+str(loss)+" %i hrs %i mins %.2f secs"%(step_hrs,step_mins,step_secs)+"\n")
		if loss <= 0.3:
			break
		else:
			continue
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
	start = time.time()
	#for tweet in range(5):
	for tweet in range(len(test_data)):
		valid_words_indices = []
		#print(tweet)
		for word in range(len(test_text[tweet].split(' '))):	
			#if test_text[tweet].split(' ')[word] not in vocab_dict:
			#	print(test_text[tweet].split(' ')[word])
			if test_text[tweet].split(' ')[word] in vocab_dict:
				#print(test_text[tweet+26].split(' ')[word])
				valid_words_indices.append(vocab_dict[test_text[tweet].split(' ')[word]])
		if len(valid_words_indices) == 0:
			pred_hc.append(0.5)
			pred_dt.append(0.5)
		if len(valid_words_indices) != 0:
			valid_words_indices = np.asarray(valid_words_indices).reshape(len(valid_words_indices),1)
			temp = sess.run(prediction, feed_dict={indices: valid_words_indices})
			pred_hc.append(temp[0][0])
			pred_dt.append(temp[0][1])
		#logits_test = sess.run([logits], feed_dict={X: matrix_test})
		#store_logits.append(logits_test) # yhats for all test tweet
	end = time.time()
	ttl = end-start
	hrs = 0
	mins = (ttl)/60
	if mins > 60:
		hrs = mins/60
		mins %= 60
	secs = (ttl)%60
	print("predicting the data finished!\n")
	print("time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))	
	sys.stdout = open("preds_test.txt","w")
	for i in range(len(pred_hc)):
		print(pred_hc[i],",",pred_dt[i])

sys.stdout = sys.__stdout__