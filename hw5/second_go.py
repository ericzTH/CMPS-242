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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.name != 'posix':
	os.chdir("M:\Course stuff\Fall 17\CMPS 242\hw5")
train_data = pd.read_csv("train.csv",header = None)
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':[1,0],'DT':[0,1]})
test_data = pd.read_csv("test.csv")

# Training Parameters
learning_rate = 0.01
training_steps = 500
batch_size = 1

text = train_data[0].copy()
test_text = test_data['tweet'].copy()

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
vocab_dict['Increíble.\nhttps://t.co/PmerodqGzQ'] = vocab_dict['Increíble']
vocab_dict['Idaho:'] = vocab_dict['Idaho']
vocab_dict['Hawaii:\nhttps://t.co/MnIlk2l9hP\nIdaho:\nhttps://t.co/7y5RxLpZRQ\nMississippi:\nhttps://t.co/n43cPeJIqa\nMichigan:\nhttps://t.co/GL5JiZbqIc'] = vocab_dict['Hawaii']
vocab_dict['Mississippi:'] = vocab_dict['Mississippi']
vocab_dict['Michigan'] = vocab_dict['Michigan']
vocab_dict['#VoteTrumpMS!'] = vocab_dict['Trump']
vocab_dict['#Trump2016'] = vocab_dict['Trump']
vocab_dict['#NeverForget\nhttps://t.co/G5TMAUzy0z'] = vocab_dict['Trump']
vocab_dict['#VoteTrumpID!'] = vocab_dict['Trump']
vocab_dict['#VoteTrumpHI!'] = vocab_dict['Trump']
vocab_dict['#VoteTrumpMI!'] = vocab_dict['Trump']
vocab_dict['#VoteTrumpMS!'] = vocab_dict['Trump']
vocab_dict['Trump!'] = vocab_dict['Trump']
vocab_dict['trump'] = vocab_dict['Trump']
vocab_dict['Presidential.'] = vocab_dict['Presidential']
vocab_dict['#WheresHillary?'] = vocab_dict['Trump']
vocab_dict['#MakeAmericaGreatAgain\n#Trump2016\xa0https://t.co/awow5pyn7n'] = vocab_dict['Trump']
vocab_dict['today.'] = vocab_dict['today']


h_iden = np.identity(len(vocab_dict))
print("size of TfIdf vocabulary(Number of Unique Words from the data set): ",len(vocab_dict),
	"\nTotal number of tweets: ",len(text),"\nTotal tweets in test: ",len(test_text))

##LSTM MODEL

# creating the model
tf.reset_default_graph()
# Network Parameters
num_input = len(vocab_dict) # number of unique words
#timesteps = 32 # timesteps
num_hidden = 256 # LSTM Hidden Layer size
hidden_unit_size = 64 # Feed Forward NN Hidden Layer size
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


def RNN(indices,weights,biases):
	#print(tf.shape(indices))
	embedding_mat = tf.get_variable('embedding_matrix',[len(vocab_dict),embedding_dims])
	x = tf.nn.embedding_lookup(embedding_mat,indices) #,[1,tf.shape(indices),embedding_dims])
	lstm_cell = tf.contrib.rnn.GRUCell(num_hidden)#, forget_bias=1.0)
	#x = tf.transpose(x)	
	# Get lstm cell output
	outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# relu activation for ff nn hidden layer
	print("shape of outputs",outputs.shape)
	hidden_layer = tf.nn.dropout(tf.nn.elu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l']),keep_prob = 0.6)
	#hidden_layer = tf.nn.elu(tf.matmul(outputs[-1][:],weights['h_l']) + biases['h_l'])
	# final sigmoidal output (yhat)
	yhat = tf.reshape(tf.reduce_sum(tf.nn.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out']),axis = 0),[1,2])
	return (yhat)	
embedding_dims = 100
# graph inputs

with tf.name_scope('inputs'):
	indices = tf.placeholder(tf.int32,shape=[None,1])
	Y = tf.placeholder("float", [1, num_classes])
	#last_word_in_tweet = tf.Variable(0)
with tf.name_scope('xEntropy'):
	logits = RNN(indices,weights,biases)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= Y))
	prediction = tf.nn.softmax(logits)
	#prediction = tf.nn.softplus(logits)
with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)		
	train_op = optimizer.minimize(loss_op)

with tf.name_scope('Accuracy'):
	correct_pred =  tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


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
	writer = tf.summary.FileWriter('logs', graph = tf.get_default_graph())
	for step in range(training_steps):
	#for step in range(20):
		
		t = 0
		ttl_loss = 0
		failed_tweets = 0
		step_start = time.time()
		#for tweet in range(10):
		for tweet in range(len(train_data)):
			#print(tweet)
			valid_words_indices = []
			for word in range(len(text[tweet].split(' '))):
				if text[tweet].split(' ')[word] in vocab_dict:
					valid_words_indices.append(vocab_dict[text[tweet].split(' ')[word]])
			
			if len(valid_words_indices) == 0:
				print(tweet)
				failed_tweets += 1
			
			if len(valid_words_indices)!=0:
				valid_words_indices = np.asarray(valid_words_indices).reshape(len(valid_words_indices),1)
				_,loss, acc, summary_loss = sess.run([train_op,loss_op, accuracy,summary_op2], feed_dict={indices: valid_words_indices,Y: np.asarray(labels_train[0][tweet]).reshape(1,2)})	
			#print(acc)
			#print(pred[0][0],pred[0][1])
			ttl_loss += loss
			if acc == 1 :
				t+=1			
			#if :
			#	pass tweet%900==0:
			#	print("For tweet "+str(tweet)+" which was said by"+str(labels_train[0][tweet])+"\nprediction of HC: "+str(pred[0][0])+" prediction of DT: "+str(pred[0][1]))	
		#print("\n matrix1 shape\n",matrix1.shape)	
		writer.add_summary(summary_loss,tweet)
		ttl_loss/=len(train_data)
		step_end = time.time()
		step_ttl = step_end-step_start
		step_hrs = 0
		step_mins = (step_ttl)/60
		if step_mins > 60:
			step_hrs = mins/60
			step_mins %= 60
		step_secs = (step_ttl)%60
		print("Training step "+str(step+1)+" acc: %f"%(t/(tweet+1))+"\tLoss: "+"%.4f"%(ttl_loss)+" %i hrs %i mins %.2f secs"%(step_hrs,step_mins,step_secs)+"\n")
		#print("\nFailed number:\n ",failed_tweets)
		if ttl_loss <= 0.31 or t/len(train_data) >= 0.95:
			break;
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
			pred_hc.append(0.35)
			pred_dt.append(0.65)

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
	sys.stdout = open("sg_large_emb.txt","w")
	print("id,HillaryClinton,realDonaldTrump")
	for i in range(len(pred_hc)):
		print(i,",",pred_hc[i],",",pred_dt[i])

sys.stdout = sys.__stdout__