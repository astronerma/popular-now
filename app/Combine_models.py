import re
import time
import urllib2
import datetime
import math
import random
import nltk
import sqlite3
#from twitter_tokenizer import *
import string
import numpy as np
from sklearn.utils import shuffle
from my_classes import Author,Tweet,Database,TwitterApi
from nltk.corpus import stopwords
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle


def prepare_content(content):
	stop = stopwords.words('english')
	text = content
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	word_list = text.translate(remove_punctuation_map).lower().split()
	word_dict = {}
	for w in word_list:
		#if w not in stop and not re.match(r'^http',w) and len(w)>3 and w in most_frequent_words(topic):
		if w not in stop and not re.match(r'^http',w) and len(w)>3:
			word_dict[w] = True
	return word_dict


def combined_model(columns,topic):
	# Columns refer to the RF model!
	random.seed(12345)
	random_state = np.random.RandomState(1)
	database = Database("twitter")
	database.change_cursor()
	
	# Get data from database
	data = np.array(database.get_data_for_combined_model(columns,topic))
	class0 = 0
	class1 = 0
	for item in data:
		if item['metric1'] == 0.0:
			class0 += 1.
		else:
			class1 += 1.

	# shuffle data:
	data = shuffle(data, random_state=random_state)
	# select sets
	triple = int(len(data) / 3)
	train, valid, test = data[:triple], data[triple:2*triple], data[2*triple:]

	# Train NB on train1
	features = []
	for item in train:
		text = item['content']
		cl = item['metric1']
		features.append((prepare_content(text),cl))		

	valid_features = []
	for item in valid:
		text = item['content']
		cl = item['metric1']
		valid_features.append(prepare_content(text))		

	test_features = []
	for item in test:
		text = item['content']
		cl = item['metric1']
		test_features.append(prepare_content(text))		

	#print features[0]
	#print valid_features[0]
	#print test_features[0] 
	#print len(features)

	print "Training Naive Bayes on training set"
	classifierNB = nltk.NaiveBayesClassifier.train(features)
	
	# predicting on validation and test set
	pNB_valid = []
	for f in valid_features:
		pNB_valid.append(classifierNB.classify(f))

	pNB_test = []
	for f in test_features:
		#pNB_test.append([classifierNB.prob_classify(f)['_prob_dict'][0.0],classifierNB.prob_classify(f)['_prob_dict'][1.0]])
		pNB_test.append(classifierNB.classify(f))
		

	# ------------------------------
	# Random forest
	print "Training Random Forest"

	# Now apply RF on train 2
	features = []
	target = []
	for item in train:
		row = []
		for c in columns[1:]:
			row.append(item[c])
		features.append(row)
		target.append(item[columns[0]])

	valid_features = []
	valid_target = []
	for item in valid:
		row = []
		for c in columns[1:]:
			row.append(item[c])
		valid_features.append(row)
		valid_target.append(item[columns[0]])

	test_features = []
	test_target = []
	for item in test:
		row = []
		for c in columns[1:]:
			row.append(item[c])
		test_features.append(row)
		test_target.append(item[columns[0]])

	#print features[0]
	#print valid_features[0]
	#print test_features[0]

	classifierRF = RandomForestClassifier(max_features=4,n_estimators=100,oob_score=True,compute_importances=True)
	classifierRF.fit(features, target)
	pRF_valid = classifierRF.predict(valid_features)
	pRF_test = classifierRF.predict(test_features)


	#print len(pRF_valid),len(pNB_valid)
	print "Combining predictions"
	# Combining predictions
	# validation set
	valid_class = []
	for i in range(len(pRF_valid)):
		if pRF_valid[i] == 1. and pNB_valid[i] == 1.:
			valid_class.append(1.)
		else:
			valid_class.append(0.)

	# test set
	test_class = []
	for i in range(len(pRF_test)):
		if pRF_test[i] == 1. and pNB_test[i] == 1.:
			test_class.append(1.)
		else:
			test_class.append(0.)

	fpr, tpr, thresholds = roc_curve(valid_target,valid_class)
	roc_auc = auc(fpr, tpr)
	print "AUC valid",roc_auc
	print fpr, tpr

	fpr, tpr, thresholds = roc_curve(test_target,test_class)
	roc_auc = auc(fpr, tpr)
	print "AUC test",roc_auc
	print fpr, tpr

	print "class0",class0
	print "class1",class1
	print "1/0",class1/class0

	#for i in range(len(pRF_test)):
	#	print pRF_test[i],pNB_test[i]
	#print test_class

columns = Tweet("data science").random_forest_columns()
#print columns
combined_model(columns,"data science")













