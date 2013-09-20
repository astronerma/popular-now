from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,classification_report,confusion_matrix,accuracy_score
from sklearn import cross_validation
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_curve
import pylab as pl
import pickle
from my_classes import Author,Tweet,Database,TwitterApi
import statsmodels.api as sm

def do_one_hot(data_frame,columns,column):
	dummy_nlinks = pd.get_dummies(data_frame[column])
	dummy_nlinks.columns = [column+ str(int(c)) for c in dummy_nlinks]
	#level = column+str(levels_to_keep)
	#d2 = data_frame.join(dummy_nlinks.ix[:,:level])
	d2 = data_frame.join(dummy_nlinks)
	return d2


#http://blog.yhathq.com/posts/logistic-regression-and-python.html

def fit_logistic_regression(topic,columns):
	random_state = np.random.RandomState(0)

	# Get data from database
	database = Database("twitter")

	# Data for tests in R
	#t = re.sub(" ","_",topic)
	#filename = "../../Data/new_train_"+topic+".csv"
	#log_columns = Tweet("data science").logistic_regression_columns()
	#database.save_to_csv(log_columns,topic,filename)

	# Get data from database
	# Only data older than 4 days and only non RTs
	data = database.get_data_for_fitting(columns,topic)
	
	# Prepare features and target (class variable)
	# Target variable should be first in column list!!!!
	tweets = np.array(data)
	features = tweets[:,1:]
	target = tweets[:,:1]
	n_samples, n_features = features.shape
	# Shuffle and split training and test sets
	X, y = shuffle(features, target, random_state=random_state)

	X = pd.DataFrame(features)
	X.columns = columns[1:]
	y = pd.DataFrame(target)

	#X = do_one_hot(X,columns,'nlinks')
	#X = do_one_hot(X,columns,'nat')
	#X = do_one_hot(X,columns,'nhash')
	#X = do_one_hot(X,columns,'day')
	#X.pop('nlinks')
	#X.pop('nat')
	#X.pop('nhash')
	#X.pop('day')

	triple = int(n_samples / 3)
	#print triple
	X_train, X_valid, X_test = X.ix[:triple,:], X.ix[triple:2*triple,:], X.ix[2*triple:,:]
	y_train, y_valid, y_test = y.ix[:triple,:], y.ix[triple:2*triple,:], y.ix[2*triple:,:]


	all_columns = X.columns
	best_columns = ['nat']
	#all_columns.remove('nlinks0')
	#all_columns.remove('nat0')

	max_auc = 0
	for c in all_columns:
		best_columns.append(c)
		fX_train = X_train[best_columns]
		fX_valid = X_valid[best_columns]
		#print len(fX_train),len(y_train)
		#print len(fX_valid),len(y_valid)

		#print fX_train.head()

		classifier = linear_model.LogisticRegression(C=1e5)
		classifier.fit(fX_train, y_train)

		probas = classifier.predict_proba(fX_valid)
		fpr, tpr, thresholds = roc_curve(y_valid, probas[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC test",c,roc_auc
		
		if roc_auc > max_auc:
			max_auc = roc_auc
			max_feature = c

		best_columns.pop()

	print "Best:", max_feature, max_auc


	#best_columns = ['words',"length","nlinks","nat",
	#		"nhash","mean_word","upper2lower","max_retweets",
	#		"followers","median_retweets","digits","day",
	#		"question","max_favorites","retweets_per_tweet",
	#		"median_favorites","sum_retweets","ntweets","alltweets","friends","fame"]
	
	fX_train = X_train[best_columns]
	fX_valid = X_valid[best_columns]
	classifier = linear_model.LogisticRegression(C=1e5)
	classifier.fit(fX_train, y_train)

	coef = classifier.coef_
	for i in range(len(coef[0])):
		#if abs(coef[0][i]) > 0.05: 
		print i,best_columns[i],coef[0][i]


		#print pp
		#print y_test.head()
		


# --------------------------------
columns = Tweet("data science").logistic_regression_columns()
#print columns
#fit_logistic_regression("data science",columns)
fit_logistic_regression("celebrity",columns)
#fit_logistic_regression("sport",columns)





