from __future__ import division, print_function
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import sys,os,time,re,itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# csv loader
def loader(filename):
	fname = str(filename)
	if fname == "test.csv":
		return(pd.read_csv(fname))
	else:
		return(pd.read_csv(fname,header = None))

tweets = loader("train.csv") 								#load train data as tweets
test_tweets = loader("test.csv") 							#load test data as tweets
labels = loader("labels_train_tweets.csv")[0].map({'HC':0,'DT':1})  					#load labels
						#map label names to class

def data_copy(matrix):
	if str(matrix) == 'tweets':
		return matrix.copy(tweets[0])
	if str(matrix) == 'test_tweets':
		return matrix['tweets'].copy()
	else:
		return matrix.copy()

train = data_copy(tweets)
test = data_copy(test_tweets['tweet'])
# now we have matrices with tweets from training and test data to play around without
# hurting the original files

#removing the fucking annoying links
for i in range(len(train)):
	train[0][i] = re.sub(r"http\S+", "", train[0][i]).strip()

for i in test:
	i = re.sub(r"http\S+", "", i).strip()


#for svms etc ##later copy paste svm_logreg_etc.py
vectorizer = TfidfVectorizer(lowercase = False)
vectorized_train = vectorizer.fit_transform(train[0]).toarray()
train_vocab = vectorizer.vocabulary_
vectorized_test = vectorizer.fit_transform(test).toarray()
pca = PCA(n_components = 450)
vectorized_train = pca.fit_transform(vectorized_train)
vectorized_test = pca.fit_transform(vectorized_test)

#own tokenization and preprocess
from nltk.tokenize import TweetTokenizer
lexicon = set()
for i in range(len(train)):
	lexicon.update(TweetTokenizer().tokenize(train[0][i]))
token_list = list(lexicon)

#creating a dictionary for the tokenized words
vocab_dict = {}
for i in range(len(token_list)):
	if token_list[i] not in vocab_dict:
		vocab_dict[token_list[i]] = i
vocab_dict['election:'] = vocab_dict['election']
vocab_dict['today.'] = vocab_dict['today']
vocab_dict['Increíble.\nhttps://t.co/PmerodqGzQ'] = vocab_dict['Increíble']
vocab_dict['Idaho:'] = vocab_dict['Idaho']
vocab_dict['Trump!'] = vocab_dict['Trump']
vocab_dict['trump'] = vocab_dict['Trump']
vocab_dict['Mississippi:'] = vocab_dict['Mississippi']
vocab_dict['Michigan:'] = vocab_dict['Michigan']
vocab_dict['#NeverForget\nhttps://t.co/G5TMAUzy0z'] = vocab_dict['Trump']
vocab_dict['Presidential.'] = vocab_dict['Presidential']
vocab_dict['#WheresHillary?'] = vocab_dict['Trump']
vocab_dict['#MakeAmericaGreatAgain\n#Trump2016\xa0https://t.co/awow5pyn7n'] = vocab_dict['#MakeAmericaGreatAgain']
vocab_dict['#MakeAmericaGreatAgain'] = vocab_dict['MakeAmericaGreatAgain']

max_len = max([len(tweets) for tweets in train[0]])
#zero padding
tweets = []
for i in range(len(train)):
	words_in_tweet = [0]*max_len
	for j in range(len(train[0][i].split(' '))):
		if train[0][i].split(' ')[j] in vocab_dict:
			words_in_tweet[j] = vocab_dict[train[0][i].split(' ')[j]]
	tweets.append(words_in_tweet)

np_tweets = np.asarray(tweets)
def next_batch(batch_size,matrix,labels):
	indices = np.arange(0,len(matrix))
	np.random.shuffle(indices)
	indices = indices[:batch_size]
	batch_data = matrix[indices]
	batch_labels = np.asarray(labels)[indices]
	return(batch_data,batch_labels)

#print(len(next_batch(20,np_tweets,labels)[0]))

tf.reset_default_graph()
embedding_dims = 256 #10 said Ehsan
num_input = len(vocab_dict) # number of unique words
num_hidden = 256 #20 was used before # LSTM Hidden Layer size
hidden_unit_size = 8 # Feed Forward NN Hidden Layer size
num_classes = 1 # neural network output layer
batch_size = 200
training_steps = 100
with tf.name_scope('ffnn'):
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

#graph inputs
with tf.name_scope('graph_inputs'):
	x = tf.placeholder(tf.int32,[None,np_tweets.shape[1]])
	y = tf.placeholder(tf.int32,[None,1])
	embedding_mat = tf.get_variable('embedding_matrix',[len(vocab_dict),embedding_dims])
	rnn_inputs = tf.nn.embedding_lookup(embedding_mat,x)
#LSTM Cell
with tf.name_scope('rnn'):
	lstm_cell = tf.contrib.rnn.GRUCell(num_hidden)
	outputs,state = tf.nn.dynamic_rnn(lstm_cell,rnn_inputs,dtype = tf.float32)
	val = tf.nn.dropout(outputs,0.8)
	val = tf.transpose(val,[1,0,2])
	last = tf.gather(val,int(val.get_shape()[0])-1)
	hidden_layer = tf.nn.dropout(tf.nn.elu(tf.matmul(last,weights['h_l']) + biases['h_l']),keep_prob = 0.8)
	yhat = tf.sigmoid(tf.matmul(hidden_layer,weights['out'])+biases['out'])
with tf.name_scope('xEntropy'):
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = yhat,labels = y))
	prediction = tf.nn.softmax(yhat)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
	train_op = optimizer.minimize(loss_op)
with tf.name_scope('Accuracy'):
	correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
summary_op = tf.summary.merge([tf.summary.scalar("loss",loss_op),tf.summary.scalar("Clinton",prediction[0]),tf.summary.scalar("Trump",1-prediction[0])])
sess = tf.Session()
init = tf.global_variables_initializer()

batch = int(len(np_tweets)/batch_size)
with sess.as_default():
	sess.run(init)
	loss,acc = 0,0
	for i in range(training_steps):
		
		for j in range(batch):
			batch_x,batch_y = next_batch(batch_size,np_tweets,labels)
			#print(batch_y)
			writer = tf.summary.FileWriter('logs', graph = tf.get_default_graph())
			t,l,p,a =sess.run([train_op,loss_op,prediction,accuracy],feed_dict={x: batch_x,y:np.asarray(batch_y).reshape(len(batch_y),1)})
			loss += l
			acc += a
		print(loss,acc)
