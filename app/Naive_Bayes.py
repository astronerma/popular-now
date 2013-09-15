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
from my_classes import  Author,Tweet,Database,TwitterApi
from nltk.corpus import stopwords
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import pickle
from sklearn.metrics import roc_curve, auc

ALL_WORDS = []
WORD_FEATURES = []
DOCUMENTS = []

def most_frequent_words(topic):
	# Get data from database
	database = Database("twitter")
	database.change_cursor()
	stop = stopwords.words('english')

	# Data for tests in R
	#database.save_to_csv(columns,topic,"../../Data/new_train.csv")

	# Get data from database
	# Only data older than 4 days and only non RTs

	all_words = []
	data = database.get_data_for_NB_fitting(topic)
	for item in data:
		text = item['content']
		cl = item['metric1']
		remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
		word_list = text.translate(remove_punctuation_map).lower().split()
		word_dict = {}
		for w in word_list:
			if w not in stop and not re.match(r'^http',w) and len(w)>2:
				all_words.append(w)

	most_frequent  = nltk.FreqDist(all_words)
	return most_frequent.keys()[:2000]


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


def fit_naive_bayes(topic):
	global ALL_WORDS, WORD_FEATURES, DOCUMENTS
	# Get data from database
	random.seed(12345)
	database = Database("twitter")
	database.change_cursor()
	
	# Get data from database
	# Only data older than 4 days and only non RTs
	features = []
	data = database.get_data_for_NB_fitting(topic)
	for item in data:
		text = item['content']
		cl = item['metric1']
		features.append((prepare_content(text),cl))		
	
	random.shuffle(features)
	
	#print features
	
	triple = int(len(features) / 3)
	X_train, X_valid, X_test = features[:triple], features[triple:2*triple], features[2*triple:]
	classifier = nltk.NaiveBayesClassifier.train(X_train)
	print nltk.classify.accuracy(classifier, X_valid)
	print nltk.classify.accuracy(classifier, X_test)
	

	# Save classifier
	t = re.sub(" ","_",topic)
	filename = "./models/naive_bayes_model_" + t +".p"
	print "Saving NB model to ",filename
	pickle.dump( classifier, open( filename, "wb" ) )
	

	pNB_valid = []
	valid_target = []
	for f in X_valid:
		pNB_valid.append(classifier.classify(f[0]))
		valid_target.append(f[1])

	fpr, tpr, thresholds = roc_curve(valid_target,pNB_valid)
	roc_auc = auc(fpr, tpr)
	print "AUC valid",roc_auc
	#print fpr, tpr

	pNB_test = []
	test_target = []
	for f in X_test:
		pNB_test.append(classifier.classify(f[0]))
		test_target.append(f[1])

	fpr, tpr, thresholds = roc_curve(test_target,pNB_test)
	roc_auc = auc(fpr, tpr)
	print "AUC test",roc_auc
	#print fpr, tpr


	#t = prepare_content(unicode("I don't know what to do with my project"))
	#print t
	#print classifier.classify(t)



#fit_naive_bayes("data science")
fit_naive_bayes("sport")
fit_naive_bayes("celebrity")

#b = most_frequent_words("data science")
#print b
#print len(b)










