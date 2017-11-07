'''
Created on 20 Oct, 2017

@author: Tianyi Luo

'''
import csv
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import RSLPStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from numpy import *
import time
from collections import OrderedDict
import sys
from sklearn.model_selection import KFold
from numpy.linalg import pinv

reload(sys)
sys.setdefaultencoding('utf-8')

def preprocess(readfilename, writefilename):
    print "Preprocessing..."
    reader = csv.reader(open(readfilename))
    writer = open(writefilename,'wb')
    line_num = 0
    reader.next()
    labels = []
    #test_labels = []
    
    for row in reader:
        line_num += 1
        #print line_num
        if line_num % 500 == 0:
            print line_num
        temp_label = row[0]
        temp_text = row[1]
        #get the train label list
        if temp_label == 'spam':
            labels.append(1)
        else:
            labels.append(0)
        #Make the words to lower format and Remove the stopwords
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(temp_text.decode('latin-1'))
        words_lower = [w.lower() for w in words]
        words_lower_filter_stopwords = []
        for w in words_lower:
            if w not in stopWords:
                words_lower_filter_stopwords.append(w)
        #print words_lower_filter_stopwords
        word_num = 0
        temp_sentence = ""
        for temp_word in words_lower_filter_stopwords:
            word_num += 1
            if word_num == 1:
                temp_sentence += temp_word
            else:
                temp_sentence += " " + temp_word
        temp_sentence += "\n"
        writer.write(temp_sentence)
    writer.close()
    print "Preprocessing is done!"
    return labels

def generate_tfidf_features(readfilename, writefilename):
    print "Generating tfidf features..."
    labels = preprocess(readfilename, writefilename)
    vectorized = CountVectorizer(min_df=3,decode_error='ignore')
    vectors_matrix = vectorized.fit_transform(open(writefilename))
    vocabulary = vectorized.vocabulary_
    transformer = TfidfTransformer(smooth_idf = False, norm='l2')
    tfidf_array = transformer.fit_transform(vectors_matrix)
    print tfidf_array.toarray().shape[1]
    labels_array = array(labels)
    print "Generating tfidf features is done!"
    return tfidf_array, labels_array, vocabulary

def generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val):
    vectorized = CountVectorizer(decode_error='ignore', vocabulary=train_vocabulary)
    vectors_matrix = vectorized.fit_transform(open(writefilename_val))
    transformer = TfidfTransformer(smooth_idf = False, norm='l2')
    tfidf_array = transformer.fit_transform(vectors_matrix)
    val_tfidf_array = tfidf_array
    return val_tfidf_array

def gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, ylabels_array, iterations_num, log_gd_writer_filename, throld):
    #Update the weights once using all the data points.
    t_start_gd = time.time()
    log_gd_writer = open(log_gd_writer_filename,'wb')
    print "Gradient descent training..."
    log_gd_writer.write("Gradient descent training...\n")
    update_theta = initial_theta 
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        total_gradient = 0.0
        last_theta = update_theta
        for index_traindata in range(len_traindata):
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            t_current = ylabels_array[index_traindata]
            temp_w_x = np.inner(update_theta, x_current_array)
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
            single_gradient = temp_diff_y_t * x_current_array
            total_gradient += single_gradient
        total_gradient += 2.0 * lambda_value * update_theta #for gradient descent
        #total_gradient += lambda_value #for 1-norm gradient descent
        update_theta = update_theta - update_yida * total_gradient
        if i % 20 == 0:
            current_cost = cost_value(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_gd_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_gd_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "Gradient descent training is done!"
            log_gd_writer.write("Gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_gd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_gd = time.time()
            print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
            log_gd_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
            log_gd_writer.close()
            return update_theta
        
    print "Gradient descent training is done!"
    log_gd_writer.write("Gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_gd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
    t_end_gd = time.time()
    print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
    log_gd_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
    log_gd_writer.close()
    return update_theta

def gradient_descent_nopenalty_bias(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, ylabels_array, iterations_num, temp_first_zero_vector, log_bias_gd_writer_filename, throld):
    #Update the weights once using all the data points. Add the bias in the cost function and do not punish the bias in the regularization item.
    t_start_bias_gd = time.time()
    log_bias_gd_writer = open(log_bias_gd_writer_filename,'wb')
    print "Bias gradient descent training..."
    log_bias_gd_writer.write("Bias gradient descent training...\n")
    update_theta = initial_theta
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        total_gradient = 0.0
        last_theta = update_theta
        temp_bias_feature_matrix = array([1.0])
        for index_traindata in range(len_traindata):
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            x_current_bias_array = np.column_stack((temp_bias_feature_matrix, x_current_array))
            t_current = ylabels_array[index_traindata]
            temp_w_x = np.inner(update_theta, x_current_bias_array)
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
            single_gradient = temp_diff_y_t * x_current_bias_array
            total_gradient += single_gradient
        total_gradient += 2 * lambda_value * np.inner(update_theta, temp_first_zero_vector)
        #total_gradient += lambda_value * update_theta
        update_theta = update_theta - update_yida * total_gradient
        if i % 20 == 0:
            current_cost = cost_value_bias(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_bias_gd_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_bias_gd_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy_bias(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "Bias gradient descent training is done!"
            log_bias_gd_writer.write("Bias gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_bias_gd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_bias_gd = time.time()
            print "The convergence time is " + str(t_end_bias_gd - t_start_bias_gd) + "seconds."
            log_bias_gd_writer.write("The convergence time is " + str(t_end_bias_gd - t_start_bias_gd) + "seconds." + "\n")
            log_bias_gd_writer.close()
            return update_theta
    
    print "Bias gradient descent training is done!"
    log_bias_gd_writer.write("Bias gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_bias_gd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
    t_end_bias_gd = time.time()
    print "The convergence time is " + str(t_end_bias_gd - t_start_bias_gd) + "seconds."
    log_bias_gd_writer.write("The convergence time is " + str(t_end_bias_gd - t_start_bias_gd) + "seconds." + "\n")
    log_bias_gd_writer.close()
    return update_theta
        
def mini_batch_gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, ylabels_array, iterations_num, batch_num, log_bgd_writer_filename, throld):
    #Update the weights once using batch_num data point.
    t_start_gd = time.time()
    log_bgd_writer = open(log_bgd_writer_filename,'wb')
    print "Batch gradient descent training..."
    log_bgd_writer.write("Batch gradient descent training...\n")
    update_theta = initial_theta
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        last_theta = update_theta       
        item_num = 0
        batch_gradient = 0.0
        for index_traindata in range(len_traindata):
            item_num += 1
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            t_current = ylabels_array[index_traindata]
            temp_w_x = np.inner(update_theta, x_current_array)
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
            single_gradient = temp_diff_y_t * x_current_array
            batch_gradient += single_gradient
            if item_num == 5:
                batch_gradient += 2.0 * lambda_value * update_theta
                update_theta = update_theta - update_yida * batch_gradient
                batch_gradient = 0.0
                item_num = 0
        batch_gradient += 2.0 * lambda_value * update_theta
        update_theta = update_theta - update_yida * batch_gradient
        if i % 20 == 0:        
            current_cost = cost_value(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_bgd_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_bgd_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "Batch gradient descent training is done!"
            log_bgd_writer.write("Batch gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_bgd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_gd = time.time()
            print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
            log_bgd_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
            log_bgd_writer.close()
            return update_theta
    print "Batch gradient descent training is done!"
    log_bgd_writer.write("Batch gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_bgd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
    t_end_gd = time.time()
    print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
    log_bgd_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
    log_bgd_writer.close()
    return update_theta

def stochastic_gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, ylabels_array, iterations_num, log_sgd_writer_filename, throld):
    #Update the weights once using every data point
    t_start_sgd = time.time()
    log_sgd_writer = open(log_sgd_writer_filename,'wb')
    print "Stochastic gradient descent training..."
    log_sgd_writer.write("Stochastic gradient descent training...\n")
    update_theta = initial_theta
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        last_theta = update_theta
        for index_traindata in range(len_traindata):
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            t_current = ylabels_array[index_traindata]
            temp_w_x = np.inner(update_theta, x_current_array)
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
            single_gradient = temp_diff_y_t * x_current_array
            single_gradient += 2.0 * lambda_value * update_theta
            update_theta = update_theta - update_yida * single_gradient
        if i % 20 == 0:
            current_cost = cost_value(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_sgd_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_sgd_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "Stochastic gradient descent training is done!"
            log_sgd_writer.write("Stochastic gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_sgd_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_sgd = time.time()
            print "The convergence time is " + str(t_end_sgd - t_start_sgd) + "seconds."
            log_sgd_writer.write("The convergence time is " + str(t_end_sgd - t_start_sgd) + "seconds." + "\n")
            log_sgd_writer.close()
            return update_theta
        
    print "Stochastic gradient descent training is done!"
    log_sgd_writer.write("Stochastic gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_sgd_writer.write("The best paramters of sgd are:" + str(update_theta) + "\n")
    t_end_sgd = time.time()
    print "The convergence time is " + str(t_end_sgd - t_start_sgd) + "seconds."
    log_sgd_writer.write("The convergence time is " + str(t_end_sgd - t_start_sgd) + "seconds." + "\n")
    log_sgd_writer.close()
    return update_theta

def irls(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, ylabels_array, iterations_num, log_irls_writer_filename, throld):
    #Chapter 4.3.3 of textbook
    t_start_gd = time.time()
    log_irls_writer = open(log_irls_writer_filename,'wb')
    print "Iterative reweighted least squares training..."
    log_irls_writer.write("Iterative reweighted least squares training...\n")
    update_theta = initial_theta 
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        total_gradient = 0.0
        last_theta = update_theta
        hessian_matrix = np.zeros((train_tfidf_array.shape[1], train_tfidf_array.shape[1]))
        y_predict_list = []
        for index_traindata in range(len_traindata):
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            t_current = ylabels_array[index_traindata]
            temp_w_x = np.inner(update_theta, x_current_array)
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
                y_predict_list.append(1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)))
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
                y_predict_list.append(1.0 / (1.0 + math.exp(-1.0 * temp_w_x)))
            single_gradient = temp_diff_y_t * x_current_array
            total_gradient += single_gradient
        y_predict_array = array(y_predict_list)
        R = np.diag((y_predict_array * (1- y_predict_array)).flatten())
        diag_lambda = np.zeros((initial_theta.shape[0],initial_theta.shape[0]))
        np.fill_diagonal(diag_lambda,lambda_value)
        hessian_xt_dot_R = train_tfidf_array.T * R
        hessian_xt_dot_R_dot_x = hessian_xt_dot_R * train_tfidf_array
        hessian_matrix += hessian_xt_dot_R_dot_x + diag_lambda
        #hessian_matrix += train_tfidf_array.T.dot(train_tfidf_array)
        
        total_gradient += 2.0 * lambda_value * update_theta
        #total_gradient += lambda_value
        update_theta = update_theta - update_yida * (pinv(hessian_matrix).dot(total_gradient.T)).T
        if i % 1 == 0:
            current_cost = cost_value(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_irls_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_irls_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "Iterative reweighted least squares training is done!"
            log_irls_writer.write("Gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_irls_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_gd = time.time()
            print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
            log_irls_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
            log_irls_writer.close()
            return update_theta
        
    print "Iterative reweighted least squares training is done!"
    log_irls_writer.write("Gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_irls_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
    t_end_gd = time.time()
    print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
    log_irls_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
    log_irls_writer.close()
    return update_theta

def egplusminus(lambda_value, initial_yida, alpha, initial_theta_plus, initial_theta_minus, train_tfidf_array, ylabels_array, iterations_num, log_eg_writer_filename, throld):
    #https://users.soe.ucsc.edu/~manfred/pubs/J36.pdf p15
    U = 1.0
    t_start_gd = time.time()
    log_eg_writer = open(log_eg_writer_filename,'wb')
    print "EG training..."
    log_eg_writer.write("EG training...\n")
    update_theta_plus = initial_theta_plus
    update_theta_minus = initial_theta_minus
    for i in range(iterations_num):
        update_yida = initial_yida * np.power(i + 1, -1.0 * alpha)
        len_traindata = train_tfidf_array.shape[0]
        total_gradient_plus = 0.0
        total_gradient_minus = 0.0
        last_theta_plus = update_theta_plus
        last_theta_minus = update_theta_minus
        
        for index_traindata in range(len_traindata):
            x_current = train_tfidf_array[index_traindata]
            x_current_array = x_current.toarray()
            t_current = ylabels_array[index_traindata]
            #x_current_plus_minus_array = np.column_stack((x_current_array, -1.0 * x_current_array))
            temp_w_x_plus = np.inner(update_theta_plus, x_current_array)
            temp_w_x_minus = np.inner(update_theta_minus, -1.0 * x_current_array)
            temp_w_x = temp_w_x_plus + temp_w_x_minus
            temp_diff_y_t = 0.0
            if temp_w_x <= 0:
                temp_diff_y_t = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x)) - 1.0 * t_current
            else:
                temp_diff_y_t = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x)) - 1.0 * t_current
            single_gradient_plus = temp_diff_y_t * x_current_array
            single_gradient_minus = temp_diff_y_t * (-1.0) * x_current_array
            total_gradient_plus += single_gradient_plus
            total_gradient_minus += single_gradient_minus
        
        total_gradient_plus += 2.0 * lambda_value * update_theta_plus
        total_gradient_minus += 2.0 * lambda_value * update_theta_minus
        last_theta = np.column_stack((last_theta_plus * np.exp(-0.0 * update_yida * total_gradient_plus), last_theta_minus * np.exp(-0.0 * update_yida * total_gradient_plus)))
        #total_gradient += lambda_value
        update_theta_plus = update_theta_plus * np.exp(-1.0 * update_yida * U * total_gradient_plus)
        update_theta_minus = update_theta_minus * np.exp(-1.0 * update_yida * U * total_gradient_minus)
        update_theta = np.column_stack((update_theta_plus, update_theta_minus))
        #print update_theta
        #update_theta = update_theta / np.linalg.norm(update_theta)
        #print update_theta
        if i % 20 == 0:
            current_cost = cost_value_eg(train_tfidf_array, ylabels_array, update_theta)
            print "The cost of " + str(i) + "th iteration is: " + str(current_cost)
            log_eg_writer.write("The cost of " + str(i) + "th iteration is: " + str(current_cost) + "\n")
            print "The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta))
            log_eg_writer.write("The norm2 distance of old w and new w is: " + str(np.linalg.norm(last_theta - update_theta)) + "\n")
            get_dataset_accuracy_eg(train_tfidf_array, ylabels_array, update_theta)
        if np.linalg.norm(last_theta - update_theta) < throld:
            print "EG training is done!"
            log_eg_writer.write("Gradient descent training is done!\n")
            print "The best paramters of gd are:" + str(update_theta)
            log_eg_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
            t_end_gd = time.time()
            print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
            log_eg_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
            log_eg_writer.close()
            return update_theta
        
    print "EG training is done!"
    log_eg_writer.write("Gradient descent training is done!\n")
    print "The best paramters of gd are:" + str(update_theta)
    log_eg_writer.write("The best paramters of gd are:" + str(update_theta) + "\n")
    t_end_gd = time.time()
    print "The convergence time is " + str(t_end_gd - t_start_gd) + "seconds."
    log_eg_writer.write("The convergence time is " + str(t_end_gd - t_start_gd) + "seconds." + "\n")
    log_eg_writer.close()
    return update_theta

def cost_value(train_tfidf_array, ylabels_array, updated_theta):
    total_errors = 0.0
    len_train_tfidf_array = train_tfidf_array.shape[0]
    for index_X in range(len_train_tfidf_array):
        error = 0.0
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        sigmoid_value_index_X = 0.0
        temp_w_x = np.inner(updated_theta, x_current_array)
        if temp_w_x <= 0:
            sigmoid_value_index_X = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            sigmoid_value_index_X = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if ylabels_array[index_X] == 1:
            if sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(sigmoid_value_index_X)
        elif ylabels_array[index_X] == 0:
            if 1.0 - sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(1.0 - sigmoid_value_index_X)
        total_errors += error
    error_final = -1.0 / len(ylabels_array) * total_errors
    return error_final

def get_dataset_accuracy(train_tfidf_array, ylabels_array, updated_theta):
    #get the performance of trainset utilizing logistic regression
    len_train_tfidf_array = train_tfidf_array.shape[0]
    item_num = 0
    train_right_num = 0
    for index_X in range(len_train_tfidf_array):
        item_num += 1
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        predict_train_prob = 0.0
        temp_w_x = np.inner(updated_theta, x_current_array)
        if temp_w_x <= 0:
            predict_train_prob = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            predict_train_prob = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if predict_train_prob >= 0.5 and ylabels_array[item_num - 1] == 1:
            train_right_num += 1
        if predict_train_prob < 0.5 and ylabels_array[item_num - 1] == 0:
            train_right_num += 1
    print "The accuracy of training dataset is: " + str(1.0 * train_right_num / len(ylabels_array)) + " (" + str(train_right_num) + "/" + str(len(ylabels_array)) + ")\n"
    return 1.0 * train_right_num / len(ylabels_array)

def cost_value_eg(train_tfidf_array, ylabels_array, updated_theta):
    total_errors = 0.0
    len_train_tfidf_array = train_tfidf_array.shape[0]
    for index_X in range(len_train_tfidf_array):
        error = 0.0
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        x_current_array = np.column_stack((x_current_array, -1.0 * x_current_array))
        sigmoid_value_index_X = 0.0
        temp_w_x = np.inner(updated_theta, x_current_array)
        if temp_w_x <= 0:
            sigmoid_value_index_X = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            sigmoid_value_index_X = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if ylabels_array[index_X] == 1:
            if sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(sigmoid_value_index_X)
        elif ylabels_array[index_X] == 0:
            if 1.0 - sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(1.0 - sigmoid_value_index_X)
        total_errors += error
    error_final = -1.0 / len(ylabels_array) * total_errors
    return error_final

def get_dataset_accuracy_eg(train_tfidf_array, ylabels_array, updated_theta):
    #get the performance of trainset utilizing logistic regression
    len_train_tfidf_array = train_tfidf_array.shape[0]
    item_num = 0
    train_right_num = 0
    for index_X in range(len_train_tfidf_array):
        item_num += 1
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        x_current_array = np.column_stack((x_current_array, -1.0 * x_current_array))
        predict_train_prob = 0.0
        temp_w_x = np.inner(updated_theta, x_current_array)
        if temp_w_x <= 0:
            predict_train_prob = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            predict_train_prob = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if predict_train_prob >= 0.5 and ylabels_array[item_num - 1] == 1:
            train_right_num += 1
        if predict_train_prob < 0.5 and ylabels_array[item_num - 1] == 0:
            train_right_num += 1
    print "The accuracy of training dataset is: " + str(1.0 * train_right_num / len(ylabels_array)) + " (" + str(train_right_num) + "/" + str(len(ylabels_array)) + ")\n"
    return 1.0 * train_right_num / len(ylabels_array)
    
def cost_value_bias(train_tfidf_array, ylabels_array, updated_theta):
    total_errors = 0.0
    len_train_tfidf_array = train_tfidf_array.shape[0]
    temp_bias_feature_matrix = array([1.0])
    for index_X in range(len_train_tfidf_array):
        error = 0.0
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        x_current_bias_array = np.column_stack((temp_bias_feature_matrix, x_current_array))
        sigmoid_value_index_X = 0.0
        temp_w_x = np.inner(updated_theta, x_current_bias_array)
        if temp_w_x <= 0:
            sigmoid_value_index_X = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            sigmoid_value_index_X = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if ylabels_array[index_X] == 1:
            if sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(sigmoid_value_index_X)
        elif ylabels_array[index_X] == 0:
            if 1.0 - sigmoid_value_index_X == 0.0:
                error = -inf
            else:
                error = math.log(1.0 - sigmoid_value_index_X)
        total_errors += error
    error_final = -1.0 / len(ylabels_array) * total_errors
    return error_final

def get_dataset_accuracy_bias(train_tfidf_array, ylabels_array, updated_theta):
    #get the performance of trainset utilizing logistic regression
    len_train_tfidf_array = train_tfidf_array.shape[0]
    predict_train_labels = []
    item_num = 0
    train_right_num = 0
    temp_bias_feature_matrix = array([1.0])
    for index_X in range(len_train_tfidf_array):
        item_num += 1
        x_current = train_tfidf_array[index_X]
        x_current_array = x_current.toarray()
        x_current_bias_array = np.column_stack((temp_bias_feature_matrix, x_current_array))
        predict_train_prob = 0.0
        temp_w_x = np.inner(updated_theta, x_current_bias_array)
        if temp_w_x <= 0:
            predict_train_prob = 1.0 - 1.0 / (1.0 + math.exp(1.0 * temp_w_x))
        else:
            predict_train_prob = 1.0 / (1.0 + math.exp(-1.0 * temp_w_x))
        if predict_train_prob >= 0.5 and ylabels_array[item_num - 1] == 1:
            train_right_num += 1
        if predict_train_prob < 0.5 and ylabels_array[item_num - 1] == 0:
            train_right_num += 1
    print "The accuracy of training dataset is: " + str(1.0 * train_right_num / len(ylabels_array)) + " (" + str(train_right_num) + "/" + str(len(ylabels_array)) + ")\n"
    return 1.0 * train_right_num / len(ylabels_array)

def train_gd(lambda_list, num_kfold, num_iterations, throld):
    gradient_method = "gd"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_yida = 50.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            ################################################################
            ###no bias gradient and you can try one norm regularization item(annotate one line and uncommnet another line)
            ################################################################
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            optimal_parameters_gd = gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, train_labels_array, iterations_num, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy(val_tfidf_array, val_labels_array, optimal_parameters_gd)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta = np.zeros(len(train_vocabulary_all))
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 50.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_gd = gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    test_result_accuracy = get_dataset_accuracy(test_tfidf_array, test_labels_array, optimal_parameters_gd)
    print test_result_accuracy
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()
    
def train_bias_gd(lambda_list, num_kfold, num_iterations, throld):
    gradient_method = "bias_gd"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_yida = 50.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            ################################################################
            ###bias gradient
            ################################################################
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            bias_initial_theta = np.zeros(len(train_vocabulary) + 1)
            temp_first_zero_vector = np.zeros(len(initial_theta) + 1) + 1.0
            temp_first_zero_vector[0] = 0.0
            optimal_parameters_gd_nopenalty_bias = gradient_descent_nopenalty_bias(lambda_value, initial_yida, alpha, bias_initial_theta, train_tfidf_array, train_labels_array, iterations_num, temp_first_zero_vector, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy_bias(val_tfidf_array, val_labels_array, optimal_parameters_gd_nopenalty_bias)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta = np.zeros(len(train_vocabulary_all))
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 50.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_bias = gradient_descent_nopenalty_bias(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    test_result_accuracy = get_dataset_accuracy(test_tfidf_array, test_labels_array, optimal_parameters_bias)
    print test_result_accuracy
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()
    
def train_mbgd(lambda_list, num_kfold, num_iterations, num_batch, throld):
    gradient_method = "mbgd"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_yida = 50.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            
            ################################################################
            ###batch gradient descent
            ################################################################
            batch_num = num_batch
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            optimal_parameters_bgd = mini_batch_gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, train_labels_array, iterations_num, batch_num, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy(val_tfidf_array, val_labels_array, optimal_parameters_bgd)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta = np.zeros(len(train_vocabulary_all))
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 50.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_bgd = mini_batch_gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    test_result_accuracy = get_dataset_accuracy(test_tfidf_array, test_labels_array, optimal_parameters_bgd)
    print test_result_accuracy
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()
    
def train_sgd(lambda_list, num_kfold, num_iterations, throld):
    gradient_method = "sgd"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_yida = 50.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            ################################################################
            ###stochastic gradient descent
            ################################################################
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            optimal_parameters_sgd = stochastic_gradient_descent(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, train_labels_array, iterations_num, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy(val_tfidf_array, val_labels_array, optimal_parameters_sgd)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta = np.zeros(len(train_vocabulary_all))
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 50.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_sgd = irls(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    test_result_accuracy = get_dataset_accuracy(test_tfidf_array, test_labels_array, optimal_parameters_sgd)
    print test_result_accuracy
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()
    
def train_irls(lambda_list, num_kfold, num_iterations, throld):
    gradient_method = "irls"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_yida = 50.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            ################################################################
            ###stochastic gradient descent
            ################################################################
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            optimal_parameters_irls = irls(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, train_labels_array, iterations_num, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy(val_tfidf_array, val_labels_array, optimal_parameters_irls)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta = np.zeros(len(train_vocabulary_all))
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 50.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_irls = irls(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    test_result_accuracy = get_dataset_accuracy(test_tfidf_array, test_labels_array, optimal_parameters_irls)
    print test_result_accuracy
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()
    
def train_eg(lambda_list, num_kfold, num_iterations, throld):
    gradient_method = "eg"
    readfilename_train_all = "train.csv"
    writefilename_train_all = "preprocessed_train"
    train_tfidf_array_all, train_labels_array_all, train_vocabulary_all = generate_tfidf_features(readfilename_train_all, writefilename_train_all)
    #test tfidf feature construction
    readfilename_test = "test.csv"
    writefilename_test = "preprocessed_test"
    test_tfidf_array, test_labels_array, test_vocabulary = generate_tfidf_features(readfilename_test, writefilename_test)
    #use training vocabulary to get tfidf of validation dataset
    test_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary_all, writefilename_test)
    #train tfidf feature construction
    validation_accuracy_list = []
    
    log_diff_lambda_filename = "log_diff_lambda_" + gradient_method
    log_diff_lambda_writer = open(log_diff_lambda_filename,'wb')
    for each_lambda in lambda_list:
        lambda_value = each_lambda
        current_lambda_val_accuracy_list = []
        log_diff_lambda_writer.write("lambda:" + str(each_lambda) + "\n")
        for current_fold in range(10):
            readfilename_train = "kvfold/" + str(current_fold + 1) + "/train.csv"
            writefilename_train = "kvfold/" + str(current_fold + 1) + "/preprocessed_train"
            train_tfidf_array, train_labels_array, train_vocabulary = generate_tfidf_features(readfilename_train, writefilename_train)
            
            #validation tfidf feature construction
            readfilename_val = "kvfold/" + str(current_fold + 1) + "/test.csv"
            writefilename_val = "kvfold/" + str(current_fold + 1) + "/preprocessed_test"
            val_tfidf_array, val_labels_array, val_vocabulary = generate_tfidf_features(readfilename_val, writefilename_val)
            #use training vocabulary to get tfidf of validation dataset
            val_tfidf_array = generate_val_tfidf_features_with_train_vol(train_vocabulary, writefilename_val)
                  
            #train the logistic regression
            #initial_theta = np.zeros(len(train_vocabulary))
            #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            #initial_theta_plus = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            #initial_theta_minus = np.random.uniform(-0.1,0.1,size=len(train_vocabulary))
            initial_theta_plus = np.random.uniform(0.0,1.0,size=len(train_vocabulary))
            initial_theta_minus = np.random.uniform(0.0,1.0,size=len(train_vocabulary))
            initial_theta_sum = np.sum(np.column_stack((initial_theta_plus, initial_theta_minus)))
            initial_theta_plus = np.divide(initial_theta_plus, initial_theta_sum)
            initial_theta_minus = np.divide(initial_theta_plus, initial_theta_minus)
            #initial_theta_plus = np.zeros(len(train_vocabulary)) + 1.0
            #initial_theta_minus = np.zeros(len(train_vocabulary)) + 1.0
            initial_yida = 0.2#100.0 / len(train_labels_array)
            alpha = 0.9
            iterations_num = num_iterations
            ################################################################
            ###stochastic gradient descent
            ################################################################
            log_writer_filename = "kvfold/" + str(current_fold + 1) + "/log_" + gradient_method
            #optimal_parameters_eg = egplusminus(lambda_value, initial_yida, alpha, initial_theta, train_tfidf_array, train_labels_array, iterations_num, log_writer_filename)
            optimal_parameters_eg = egplusminus(lambda_value, initial_yida, alpha, initial_theta_plus, initial_theta_minus, train_tfidf_array, train_labels_array, iterations_num, log_writer_filename, throld)
            val_result_accuracy = get_dataset_accuracy_eg(val_tfidf_array, val_labels_array, optimal_parameters_eg)
            current_lambda_val_accuracy_list.append(val_result_accuracy)
            
        val_acc_total = 0.0
        print current_lambda_val_accuracy_list
        for current_val_per in current_lambda_val_accuracy_list:
            log_diff_lambda_writer.write(str(current_val_per) + "  ")
            val_acc_total += current_val_per
        log_diff_lambda_writer.write("\n")
        log_diff_lambda_writer.write(str(1.0 * val_acc_total / num_kfold) + "\n")
        validation_accuracy_list.append(1.0 * val_acc_total / num_kfold)
    
    #get the optimal lambda
    lambda_num = -1
    max = 0.0
    max_index = 0
    for current_accu in validation_accuracy_list:
        lambda_num += 1
        if current_accu > max:
            max = current_accu
            max_index = lambda_num
    
    #retrain all the data points
    optimalized_lambda = lambda_list[max_index]
    initial_theta_plus = np.random.uniform(0.0,1.0,size=len(train_vocabulary_all))
    initial_theta_minus = np.random.uniform(0.0,1.0,size=len(train_vocabulary_all))
    initial_theta_sum = np.sum(np.column_stack((initial_theta_plus, initial_theta_minus)))
    initial_theta_plus = np.divide(initial_theta_plus, initial_theta_sum)
    initial_theta_minus = np.divide(initial_theta_plus, initial_theta_minus)
    #initial_theta_plus = np.zeros(len(train_vocabulary)) + 1.0
    #initial_theta_minus = np.zeros(len(train_vocabulary)) + 1.0
    #initial_theta = np.random.uniform(-0.1,0.1,size=len(train_vocabulary_all))
    initial_yida = 0.2#150.0 / len(train_labels_array_all)
    alpha = 0.9
    iterations_num = num_iterations
    lambda_value = optimalized_lambda
    log_writer_filename = "log_" + gradient_method
    log_writer_filename_final = "log_" + gradient_method + "_test_accuracy"
    log_final_writer = open(log_writer_filename_final,'wb')
    optimal_parameters_eg = egplusminus(lambda_value, initial_yida, alpha, initial_theta_plus, initial_theta_minus, train_tfidf_array_all, train_labels_array_all, iterations_num, log_writer_filename, throld)
    train_result_accuracy = get_dataset_accuracy_eg(train_tfidf_array_all, train_labels_array_all, optimal_parameters_eg)
    test_result_accuracy = get_dataset_accuracy_eg(test_tfidf_array, test_labels_array, optimal_parameters_eg)
    print train_result_accuracy
    print test_result_accuracy
    log_final_writer.write(str(train_result_accuracy) + "\n")
    log_final_writer.write(str(test_result_accuracy) + "\n")
    log_final_writer.close()

if __name__ == '__main__':
    lambda_list = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    #lambda_list =  [0.0001]
    num_kfold = 10
    num_iterations = 1000
    num_batch = 30
    throld = 0.005
    
    train_gd(lambda_list, num_kfold, num_iterations, throld)
    train_bias_gd(lambda_list, num_kfold, num_iterations, throld)
    train_mbgd(lambda_list, num_kfold, num_iterations, num_batch, throld)
    train_sgd(lambda_list, num_kfold, num_iterations, throld)
    train_irls(lambda_list, num_kfold, num_iterations, throld)
    train_eg(lambda_list, num_kfold, num_iterations, throld)
        