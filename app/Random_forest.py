from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc 
from sklearn import cross_validation
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_curve
import pylab as pl
import pickle
from my_classes import Author,Tweet,Database,TwitterApi


# http://www.stanford.edu/~stephsus/R-randomforest-guide.pdf
# http://scikit-learn.org/stable/modules/ensemble.html
# http://scikit-learn.org/stable/auto_examples/plot_roc.html

def fit_random_forest(topic,columns):
	random_state = np.random.RandomState(1)

	# Get data from database
	database = Database("twitter")

	# Data for tests in R
	t = re.sub(" ","_",topic)
	filename = "../../Data/new_train_"+topic+".csv"
	log_columns = Tweet("data science").ligistic_regression_columns()
	database.save_to_csv(log_columns,topic,filename)

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

	#half = int(n_samples / 2)
	#X_train, X_test = X[:half], X[half:]
	#y_train, y_test = y[:half], y[half:]

	triple = int(n_samples / 3)
	#print triple
	X_train, X_valid, X_test = X[:triple], X[triple:2*triple], X[2*triple:]
	y_train, y_valid, y_test = y[:triple], y[triple:2*triple], y[2*triple:]

	classifier = RandomForestClassifier(max_features=4,n_estimators=100,oob_score=True,compute_importances=True)
	#classifier = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=0)
	classifier.fit(X_train, y_train)

	t = re.sub(" ","_",topic)
	filename = "../../models/random_forest_model_" + t +".p"
	print "Saving RF model to ",filename
	pickle.dump( classifier, open( filename, "wb" ) )

	#classifier = pickle.load( open( "random_forest_model.p", "rb" ) )

	importances = classifier.feature_importances_
	
	diagnostics = True

	if diagnostics == True:
		for i in range(len(importances)):
			print columns[i+1],importances[i]
		print "-----------------------"
		#print "Datasets sizes:",len(y_train),len(y_valid),len(y_test)
		print "Datasets sizes:",len(y_train),len(y_test)
		
		# Compute ROC curve and area the curve
		probas_ = classifier.predict_proba(X_train)
		fpr, tpr, thresholds = roc_curve(y_train, probas_[:, 1])

		roc_auc = auc(fpr, tpr)
		print "AUC training",roc_auc
		#print fpr,tpr
		probas_ = classifier.predict_proba(X_valid)
		fpr, tpr, thresholds = roc_curve(y_valid, probas_[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC validation",roc_auc
		#print fpr,tpr
		probas_ = classifier.predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
		roc_auc = auc(fpr, tpr)
		print "AUC test",roc_auc
		#print fpr,tpr

	should_i_plot = False
	if should_i_plot:
		precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])
		area = auc(recall, precision)
		print("Area Under Curve: %0.2f" % area)
		pl.clf()
		pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		pl.plot([0, 1], [0, 1], 'k--')
		pl.xlim([0.0, 1.0])
		pl.ylim([0.0, 1.0])
		pl.xlabel('False Positive Rate')
		pl.ylabel('True Positive Rate')
		pl.title('Receiver operating characteristic example')
		pl.legend(loc="lower right")
		pl.show()

	#print features[0]
	#print classifier.predict_proba(features[0])
	#return fitted_model

	#def classify_tweet(columns)

# --------------------------------
columns = Tweet("data science").random_forest_columns()
#print columns
#fit_random_forest("data science",columns)
fit_random_forest("celebrity",columns)
fit_random_forest("sport",columns)





