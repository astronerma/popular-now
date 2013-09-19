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
	random_state = np.random.RandomState(230)

	print "N columns:",len(columns)

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
	features, target = shuffle(features, target, random_state=10)

	
	X = pd.DataFrame(features)
	X.columns = columns[1:]
	y = pd.DataFrame(target)



	#X = do_one_hot(X,columns,'nlinks')
	#X = do_one_hot(X,columns,'nat')
	#X = do_one_hot(X,columns,'nhash')
	#X.pop('nlinks')
	#X.pop('nat')
	#X.pop('nhash')


	triple = int(n_samples / 3)
	#print triple
	X_train, X_valid, X_test = X.ix[:triple,:], X.ix[triple:2*triple,:], X.ix[2*triple:,:]
	y_train, y_valid, y_test = y.ix[:triple,:], y.ix[triple:2*triple,:], y.ix[2*triple:,:]

	classifier = linear_model.LogisticRegression(C=1e5)
	classifier.fit(X_train, y_train)

	#rint classifier.coef_

	t = re.sub(" ","_",topic)
	filename = "../../models/logistic_regression_model_" + t +".p"
	print "Saving logistic model to ",filename
	pickle.dump( classifier, open( filename, "wb" ) )


	diagnostics = True


	# -----------------
	if diagnostics == True:
		#for i in range(len(importances)):
		#	print columns[i+1],importances[i]
		print "-----------------------"
		#print "Datasets sizes:",len(y_train),len(y_valid),len(y_test)
		print "Datasets sizes:",len(y_train),len(y_test)
		
		# Compute ROC curve and area the curve
		probas = classifier.predict_proba(X_train)
		fpr, tpr, thresholds = roc_curve(y_train, probas[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC training",roc_auc
		#print probas_
		#pp = classifier.predict(X_train)
		#print classification_report(y_test, pp.ix[:,0])


		#print fpr,tpr
		probas = classifier.predict_proba(X_valid)
		fpr, tpr, thresholds = roc_curve(y_valid, probas[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC validation",roc_auc
		#print fpr,tpr
		probas = classifier.predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC test",roc_auc
		

		#print pp
		#print y_test.head()
		
		pp = classifier.predict(X_test)
		tp = 0.
		tn = 0.
		fp = 0.
		fn = 0.
		for i in range(len(pp)):
			#print y_test.iloc[i].values[0],pp[i]
			#print pp[i]
			if y_test.iloc[i].values[0] == 0.:
				if pp[i] == 0.:
					tn += 1
				else:
					fp += 1
			else:
				if pp[i] == 1.:
					tp += 1
				else:
					fn += 1

		a = len(pp)
		print tp, fp
		print fn, tn
		print "TPF", tp/(tp+fn)
		print "FPF", fp/(tn+fp)
		print "TP/FP",tp/fp


		# --------------
		print "----------------"
		threshold = 0.3
		print "Using a threshold",threshold
		probas = classifier.predict_proba(X_test)

		tp = 0.
		tn = 0.
		fp = 0.
		fn = 0.
		for i in range(len(y_test)):
			#print y_test.iloc[i].values[0],pp[i]
			#print pp[i]
			if probas[i][1] > threshold:
				pp = 1.
			else:
				pp = 0.

			if y_test.iloc[i].values[0] == 0.:
				if pp == 0.:
					tn += 1
				else:
					fp += 1
			else:
				if pp == 1.:
					tp += 1
				else:
					fn += 1

		a = len(y_test)
		print tp, fp
		print fn, tn
		print "TPF", tp/(tp+fn)
		print "FPF", fp/(tn+fp)
		print "TP/FP",tp/fp


		print "----------"
		print "Test with labels shuffled"
		pp = shuffle(y_test)
		tp = 0.
		tn = 0.
		fp = 0.
		fn = 0.
		for i in range(len(pp)):
			#print y_test.iloc[i].values[0],pp[i]
			#print pp[i]

			if y_test.iloc[i].values[0] == 0.:
				if pp[i] == 0.:
					tn += 1
				else:
					fp += 1
			else:
				if pp[i] == 1.:
					tp += 1
				else:
					fn += 1

		a = len(pp)
		print tp, fp
		print fn, tn
		print "TPF", tp/(tp+fn)
		print "FPF", fp/(tn+fp)
		print "TP/FP",tp/fp

		#print "True positives": tp
		#print "True negatives"


		#print guessed/len(pp)

		#print fpr,tpr
		#confusion_matrix(y_test,probas_)


	should_i_plot = False
	if should_i_plot:
		precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
		area = auc(recall, precision)
		print("Area Under Curve: %0.2f" % area)
		pl.clf()
		pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		pl.plot([0, 1], [0, 1], 'k--')
		pl.xlim([0.0, 1.0])
		pl.ylim([0.0, 1.0])
		pl.xlabel('False Positive Rate',fontsize=20)
		pl.ylabel('True Positive Rate',fontsize=20)
		pl.title('Receiver operating characteristic',fontsize=20)
		pl.legend(loc="lower right")
		pl.show()

	#print features[0]
	#print classifier.predict_proba(features[0])
	#return fitted_model

	#def classify_tweet(columns)

# --------------------------------
columns_ds = Tweet("data science").logistic_regression_columns_ds()
columns_c = Tweet("data science").logistic_regression_columns_c()
columns_s = Tweet("data science").logistic_regression_columns_s()

#print columns
#fit_logistic_regression("data science",columns_ds)
#fit_logistic_regression("celebrity",columns_c)
fit_logistic_regression("sport",columns_s)





