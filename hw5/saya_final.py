
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.contrib import rnn
import os,re
import numpy as np
import pandas as pd


# In[3]:


test_data = pd.read_csv("test.csv")
print(len(test_data))
#test_data.head(10)


# In[4]:


train_data = pd.read_csv("train.csv",header = None)
print(len(train_data))
train_data.describe()
#train_data[0].head(10)


# In[5]:


text = train_data[0].copy()
test_text = test_data['tweet'].copy()
labels_train = pd.read_csv("labels_train_tweets.csv",header = None)
labels_train[0] = labels_train[0].map({'HC':1,'DT':0})
#test_text.head()
#Hillary = 1 , DT = 0


# In[6]:


#labels_train.head(10)


# In[7]:

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


# In[8]:


#The actual vocab size was around 12k, but using a larger number 
#reduces the probability of collisions from the hash function.
vocab_size = 10 #000
encoded_tweets = [one_hot(d, vocab_size) for d in train_data[0]]
encoded_test = [one_hot(d, vocab_size) for d in test_text]
tweet_lengths = [len(t) for t in encoded_tweets]
test_lengths = [len(t) for t in encoded_test]
max_length = 0
for t in encoded_tweets:
	if len(t) > max_length:
		max_length = len(t)
		
for t in encoded_test:
	if len(t) > max_length:
		max_length = len(t)


# labels_train
# padded_tweets
# tweet_lengths

# In[9]:


class SimpleDataIterator():
	def __init__(self, df):
		self.df = df
		self.size = len(self.df)
		self.epochs = 0
		self.shuffle()

	def shuffle(self):
		self.df = self.df.sample(frac=1).reset_index(drop=True)
		self.cursor = 0

	def next_batch(self, n):
		if self.cursor+n-1 > self.size:
			self.epochs += 1
			self.shuffle()
		res = self.df.ix[self.cursor:self.cursor+n-1]
		self.cursor += n
		return res['data'], res['labels'], res['length']


# In[10]:


train_dic={}
train_dic["data"] = encoded_tweets
train_dic["labels"] = labels_train[0].ravel().tolist()
train_dic["length"] = tweet_lengths
train_len = len(train_data)
test_len = len(test_data)

train = pd.DataFrame.from_dict(data=train_dic, orient='columns', dtype=None)


test_dic={}
test_dic["data"] = encoded_test
test_dic["length"] = test_lengths
test = pd.DataFrame.from_dict(data=test_dic, orient='columns', dtype=None)

test_input = test.values

data = SimpleDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])


# In[11]:


class PaddedDataIterator(SimpleDataIterator):
	def next_batch(self, n):
		if self.cursor+n > self.size:
			self.epochs += 1
			self.shuffle()
		res = self.df.ix[self.cursor:self.cursor+n-1]
		self.cursor += n

		# Pad sequences with 0s so they are all the same length
		maxlen = max(res['length'])
		x = np.zeros([n, maxlen], dtype=np.int32)
		for i, x_i in enumerate(x):
			x_i[:res['length'].values[i]] = res['data'].values[i]

		return x, res['labels'], res['length']


# In[12]:


#data = PaddedDataIterator(train)
train_data = PaddedDataIterator(test)
#d = data.next_batch(3)
#print('Input sequences\n', d[0], end='\n\n')


# In[15]:


def reset_graph():
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()

def build_graph(
	vocab_size = vocab_size,
	state_size = 24,
	batch_size = 189,
	num_classes = 2):

	reset_graph()

	# Placeholders
	x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
	seqlen = tf.placeholder(tf.int32, [batch_size])
	y = tf.placeholder(tf.int32, [batch_size])
	keep_prob = tf.placeholder_with_default(1.0, [])

	# Embedding layer
	embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

	# RNN
	cell = tf.contrib.rnn.GRUCell(state_size)
	#cell = tf.nn.rnn_cell.BasicLSTMCell(state_size,forget_bias = 1)
	#cell = tf.contrib.rnn.LSTMCell(state_size,forget_bias = 1)
	init_state = tf.get_variable('init_state', [1, state_size],
								 initializer=tf.constant_initializer(0.0))
	init_state = tf.tile(init_state, [batch_size, 1])
	rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs,dtype=tf.float32)#, sequence_length=seqlen,initial_state=init_state)

	# Add dropout, as the model otherwise quickly overfits
	rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

	"""
	Obtain the last relevant output. The best approach in the future will be to use:

		last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

	which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
	gradient for this op has not been implemented as of this writing.

	The below solution works, but throws a UserWarning re: the gradient.
	"""
	idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
	last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

	# Softmax layer
	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
	logits = tf.matmul(last_rnn_output, W) + b
	preds = tf.nn.softmax(logits)
	correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	return {
		'x': x,
		'seqlen': seqlen,
		'y': y,
		'dropout': keep_prob,
		'loss': loss,
		'ts': train_step,
		'preds': preds,
		'accuracy': accuracy
	}

def train_graph(graph, sess, batch_size = 189, num_epochs = 50000, iterator = PaddedDataIterator):
	
	
	sess.run(tf.global_variables_initializer())
	tr = iterator(train)
	te = iterator(test)

	step, accuracy = 0, 0
	tr_losses, te_losses = [], []
	current_epoch = 0
	while current_epoch < num_epochs:
		step += 1
		batch = tr.next_batch(batch_size)
		feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
		accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
		accuracy += accuracy_
		if step >1 and accuracy/step >= 0.97:
			print("Accuracy after epoch", current_epoch, " - tr:", accuracy / step)
			break;

		if tr.epochs > current_epoch:
			current_epoch += 1
			tr_losses.append(accuracy / step)
			step, accuracy = 0, 0
			#eval test set
			""" te_epoch = te.epochs
			while te.epochs == te_epoch:
				step += 1
				batch = te.next_batch(batch_size)
				feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
				accuracy_ = sess.run([g['accuracy'],g['loss']], feed_dict=feed)[0]
				accuracy += accuracy_

			te_losses.append(accuracy / step)"""
			step, accuracy = 0,0
			if current_epoch%100 == 0:
				print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1])#, "- te:", te_losses[-1])
	return tr_losses#, te_losses


# In[ ]:


g = build_graph()
#tr_losses, te_losses = train_graph(g)
sess = tf.Session()
tr_losses = train_graph(g, sess)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
x = [x for x in range(len(tr_losses))]
plt.figure(figsize=(20,15))
plt.plot(x,tr_losses, label = "Training Step vs Training Accuracy")
plt.xlabel("epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.savefig("best_LSTM.png")
plt.show()


# In[ ]:


test_input = test
test_input['data'] = test['data'].apply(lambda x : np.array(x))
test_input['length'] = test_input['length'].apply(lambda x : np.array(x))
maxlen = max(test_input['length'])
x = np.zeros([len(test_input), maxlen], dtype=np.int32)
for i, x_i in enumerate(x):
	x_i[:test_input['length'].values[i]] = test_input['data'].values[i]

results = []
import math
for i in range(math.floor(len(x)/189)):
	data_bit = x[i*189:(i+1)*189]
	len_bit = test_input['length'][i*189:(i+1)*189]
	results.append(sess.run([g["preds"]], feed_dict={g['x']: data_bit, g['seqlen'] : len_bit}))
print(results[0])


# In[ ]:


print(results[0][0][32])


# In[ ]:



with open("output_GRU.txt", "w") as f:
	f.write("id,realDonaldTrump,HillaryClinton\n")
	#batch of 189
	i = 0
	print (len(results))
	for n in range(len(results)):
		print (len(results[n][0]))
		for y in range(len(results[n][0])):
			f.write("{},{:06f},{:06f}\n".format(i, results[n][0][y][0],results[n][0][y][1]))
			i = i+1