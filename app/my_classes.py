import MySQLdb
import MySQLdb.cursors
import tweepy
import numpy as np
import math
import re
import time
import random
from numpy import median
import pickle
from nltk.corpus import stopwords
import string

# Global variables
#classifierRF = {}
#classifierRF['data science'] = pickle.load( open( "app/random_forest_model_data_science.p", "rb" ) )
#classifierRF['celebrity'] = pickle.load( open( "app/random_forest_model_celebrity.p", "rb" ) )
#classifierRF['sport'] = pickle.load( open( "app/random_forest_model_sport.p", "rb" ) )

#classifierNB = {}
#classifierNB['data science'] = pickle.load( open( "app/naive_bayes_model_data_science.p", "rb" ) )
#classifierNB['celebrity'] = pickle.load( open( "app/naive_bayes_model_celebrity.p", "rb" ) )
#classifierNB['sport'] = pickle.load( open( "app/naive_bayes_model_sport.p", "rb" ) )


####################################
# - DATABASE -----------------------

class Database:
	def __init__(self,name):
		self.name = name
		self.db =  MySQLdb.connect(host="localhost",user="root",passwd="", db=self.name ,charset='utf8mb4',use_unicode=True)
		self.coursor = self.db.cursor()

	def change_cursor(self):
		self.close_connection()
		self.db =  MySQLdb.connect(host="localhost",user="root",passwd="", db=self.name ,charset='utf8mb4',use_unicode=True,cursorclass=MySQLdb.cursors.DictCursor)
		self.coursor = self.db.cursor()
	
	def show(self):
		self.coursor.execute("SHOW TABLES;")
		#print [x[0] for x in self.coursor.fetchall()]

	def close_connection(self):
		self.db.commit()
		self.db.close()

	def query(self,query):
		self.coursor.execute(query)
		return self.coursor.fetchall()

	def select_data(self,table,columnnames):
		# Returns array of database rows for specified columns
		if columnnames == "*":
			query = "SELECT * FROM "+ table +";"
		else:
			query = "SELECT "
			for col in columnnames:
				query += col + ","
			query = query[:-1] + " "
			query += "FROM "+table+";"
		self.coursor.execute(query)
		if len(columnnames) == 1:
			return [x[0] for x in self.coursor.fetchall()]
		else:
			return self.coursor.fetchall()

	def select_data_for_author(self,author_id,table,column):
		query = "SELECT " + column + " FROM " + table + " WHERE author_id = " + str(author_id) + " and rt = 0;"
		self.coursor.execute(query)
		return [x[0] for x in self.coursor.fetchall()]

	def get_max_id(self,table,topic):
		# Returns a number which is a tweet_id
		query = "SELECT id FROM " + table + " WHERE topic = " + "\"" + topic + "\" ORDER BY id DESC LIMIT 1;"	
		self.coursor.execute(query)
		ids = self.coursor.fetchall()
		if len(ids) == 0:
			return 0
		else:
			return ids[0][0]

	# get minimum id of a tweet for a given user and topic
	def get_min_id(self,author_id,table,topic):
		# Returns a number which is a tweet_id
		query = "SELECT id FROM " + table + " WHERE author_id = "+ str(author_id) + " and topic = " + "\"" + topic + "\" ORDER BY id LIMIT 1;"	
		#print query
		self.coursor.execute(query)
		return self.coursor.fetchall()[0][0]

	def tables(self):
		query = "SHOW TABLES;"
		self.coursor.execute(query)
		return [table[0] for table in self.coursor.fetchall()]

	def tableExists(self,table):
		if table in self.tables():
			return True
		else:
			return False

	def create_table(self,table_name,columns):
		# Get column names
		query = "CREATE TABLE " + table_name +" ("
		for k in sorted(columns.keys()):
			query += k + " " + columns[k] + ","
		query = query [:-1] + ");"
		self.coursor.execute(query)
		self.db.commit()
		#print query

	def insert(self,table_name,stuff):
		# Stuff is an object, Author or Tweet
		query = "INSERT INTO " + table_name + " ("
		columns = tuple(sorted(stuff.__dict__.keys()))
		values = []
		for c in columns:
			if stuff.__dict__[c] != None:
				query += c + ","
		query = query[:-1]+ ") VALUES ("
		for c in columns:
			if stuff.__dict__[c] != None:
				query += "%s " + ","
				values.append(stuff.__dict__[c])
		query = query[:-1]+ ");"
		try:
			self.coursor.execute(query,tuple(values))
		except:
			print "Something wrong with the values"
			#print values


		self.db.commit()
		#print query,tuple(values)

	def update_author(self,table_name,stuff,author_id):
		# I AM DELETING a given author FIRST AND THEn I am INSERTING it back
		# Stuff is the new author data to put back in
		query = "DELETE FROM " + table_name + " WHERE author_id = " + str(author_id) + ";"
		self.coursor.execute(query)
		self.db.commit()
		self.insert("authors",stuff)

	def drop_table(self,table_name):
		query = "DROP TABLE "+ table_name
		self.coursor.execute(query)


	def define_sets(self,topic):
		random.seed()
		pass

	def save_to_csv(self,columns,topic,filename):
		query = 'SELECT '
		for c in columns:
			query += c + ","
		query = query[:-1] + ' FROM tweets,authors WHERE tweets.rt=0 and tweets.author_id = authors.author_id and tweets.topic = '+ "\"" + topic + "\"" + ';'
		#print query
		self.coursor.execute(query)
		tweets = self.coursor.fetchall()
		f = open(filename,"w")
		
		out = ""
		for c in columns:
			out += c + "|"
		out = out[:-1] + "\n"
		f.write(out)

		for line in tweets:
			out = ""
			for c in line:
				out += str(c) + "|"
			out = out[:-1]+"\n"
			f.write(out)

	def get_data_for_fitting(self,columns,topic):
		query = 'SELECT '
		for c in columns:
			query += c + ","
		query = query[:-1] + ' FROM tweets,authors WHERE tweets.age > 4 and tweets.age < 365 and \
		tweets.rt = 0 and tweets.author_id = authors.author_id and tweets.topic = '+ "\"" + topic + "\"" + ';'
		#print query
		self.coursor.execute(query)
		return(self.coursor.fetchall())

	def get_data_for_NB_fitting(self,topic):
		query = 'SELECT content,metric1 FROM tweets WHERE tweets.age > 4 and \
		tweets.rt = 0 and tweets.topic = '+ "\"" + topic + "\"" + ';'
		#print query
		self.coursor.execute(query)
		return(self.coursor.fetchall())

	def get_data_for_combined_model(self,columns,topic):
		self.change_cursor()
		query = 'SELECT '
		for c in columns:
			query += c + ","
		query = query[:-1] + ',content,id FROM tweets,authors WHERE tweets.age > 4 and \
		tweets.rt = 0 and tweets.author_id = authors.author_id and tweets.topic = '+ "\"" + topic + "\"" + ';'
		#print query
		self.coursor.execute(query)
		return(self.coursor.fetchall())
	


	def get_latest_tweets(self,topic,count):
		# Connect to twitter API
		api = TwitterApi()
		max_tweet = self.get_max_id("new",topic)+1
		#print max_tweet	
		tweets = api.get_list_timeline(max_tweet,topic,count)
		print "found",len(tweets),"new tweets in",topic

		for t in tweets:
			
			tweet = Tweet(topic)
			tweet.get_original_features(t)
			tweet.get_new_features()
			print "inserting"
			self.insert("new",tweet)
		
	def delete_older_tweets(self,hours):

		query = "DELETE FROM new WHERE hour(TimeDiff(now(),date)) > " + str(hours-7) + ";"
		print query 
		self.coursor.execute(query)



	def get_author_info(self,topic):
		query = "SELECT name,image_url,description,followers,retweets_per_tweet,screen_name FROM authors WHERE topic = " +"\"" + topic + "\";"
		#print query
		self.coursor.execute(query)
		authors = self.coursor.fetchall()
		return authors

	# This creates a new metric by which tweets are considered interesting or not
	# It sits in metric1 and is 1 is retweets > retweets_per_tweets
	def create_new_metric(self):
		self.change_cursor()
		q1 = 'SELECT id,retweets,retweets_per_tweet,age FROM tweets,authors WHERE tweets.author_id = authors.author_id;'
		#print q1
		self.coursor.execute(q1)
		data = self.coursor.fetchall()
		for item in data:
			if item['age'] > 4:
				if item['retweets'] >= math.ceil(item['retweets_per_tweet']):
					metric = 1.
					#print item['retweets'],math.ceil(item['retweets_per_tweet'])
				else:
					metric = 0.
			else:
				metric = 0.
			#print metric
			q2 = 'UPDATE tweets SET metric1 = %s WHERE id=%s'
			p2 = (metric,item['id'])
			self.coursor.execute(q2,p2)


	def update_all_tweets_in_new_table(self):
		self.change_cursor()
		q1 = 'SELECT id,retweets,retweets_per_tweet,age FROM new,authors WHERE new.author_id = authors.author_id \
		and new.metric2=0;'
		self.coursor.execute(q1)
		data = self.coursor.fetchall()
		for item in data:
			q3 = 'UPDATE new SET metric2 = %s WHERE id=%s'
			p3 = (item['retweets_per_tweet'],item['id'])
			self.coursor.execute(q3,p3)

	def update_class_in_new_table(self,columns,topic):
		global classifierRF, classifierNB
		
		if topic == "data science":
			columns = Tweet("data science").logistic_regression_columns_ds()[1:]
		elif topic == "celebrity":
			columns = Tweet("data science").logistic_regression_columns_c()[1:]
		elif topic == "sport":
			columns = Tweet("data science").logistic_regression_columns_s()[1:]



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
		

		# Classifies
		t = re.sub(" ","_",topic)
		#filename = "../models/random_forest_model_" + t +".p"
		#classifierRF = pickle.load( open( filename, "rb" ))
		filename = "../models/logistic_regression_model_" + t +".p"
		classifier = pickle.load( open( filename, "rb" ))
		#filename = "app/models/naive_bayes_model_" + t +".p"
		#classifierNB = pickle.load( open( filename, "rb" ))
		
		# Update links, needs to be done here to avoid alterations to content variable
		#pattern = re.compile(r"(http://t\.co/[0-9a-zA-Z/:]+)")
		
		# download not updated tweets
		self.change_cursor()
		q1 = 'SELECT * FROM new,authors WHERE new.author_id = authors.author_id \
		and new.metric2=99. and new.topic = '+ '\"' + topic + '\";'
		self.coursor.execute(q1)
		data = self.coursor.fetchall()
		#print q1
		for row in data:		
			features = []
			for c in columns:
				features.append(row[c])
		
			#myclassRF = classifierRF.predict(features)[0]
			if classifier.predict_proba(features)[0][1] > 0.4:
				myclass = 1.0
			else:
				myclass = 0.0


			if row["retweets"] > row["retweets_per_tweet"]+1:
				myclass = 2.0

			#myclass = 5.0
			#row['content'] = pattern.sub("<a href=\"\\1\" target=\"_blank\"> <i class=\"icon-hand-right\"></i> link </a>",row['content'])
			q3 = 'UPDATE new SET metric2 = %s WHERE id=%s'
			try:
				p3 = (myclass,row['id'])
			except:
				print "Sorry the content was too long!"
			#print q3,p3
			self.coursor.execute(q3,p3)



	def classify_NB(table,self):
		# This function will use NB prediction on all tweets in tweets table!
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
		
		# Read my classifier
		t = re.sub(" ","_",topic)
		filename = "app/naive_bayes_model_" + t +".p"
		classifier = pickle.load( open( filename, "rb" ) )

		self.change_cursor()
		q1 = 'SELECT id,content FROM ' + table +';'
		self.coursor.execute(q1)
		data = self.coursor.fetchall()
		for item in data:
			text = prepare_content(item['content'])
			metric2 = classifier.predict(text)
			q3 = 'UPDATE new SET metric2 = %s WHERE id=%s'
			p3 = (metric2,item['id'])
			self.coursor.execute(q3,p3)

	def get_names(self):
		self.change_cursor()
		q1 = 'SELECT name,screen_name FROM authors;'
		self.coursor.execute(q1)
		data = self.coursor.fetchall()
		namedict = {}
		for n in data:
			namedict[n['screen_name']] = n['name']
		return namedict



####################################
# - AUTHOR -------------------------

class Author:
	def __init__(self,topic):
		self.author_id = None
		self.name = None
		self.image_url = None
		self.description = None
		self.screen_name = None
		self.topic = topic
		self.friends = None
		self.ntweets = None
		self.alltweets = None
		self.fame = None
		self.listed = None
		self.zone = None

		self.sum_retweets = None
		self.sum_favorites = None
		self.followers = None
		self.max_retweets = None
		self.max_favorites = None
		self.median_retweets = None
		self.median_favorites = None
		self.retweets_per_tweet = None
		self.favorites_per_tweet = None


	def get_original_features(self,profile_from_api):
		self.author_id = profile_from_api.id
		self.name = profile_from_api.name
		self.image_url = profile_from_api.profile_image_url
		self.description = profile_from_api.description
		self.screen_name = profile_from_api.screen_name
		self.followers = profile_from_api.followers_count
		self.alltweets = profile_from_api.statuses_count
		self.friends = profile_from_api.friends_count
		self.fame = self.followers/self.friends
		self.listed = profile_from_api.listed_count
		
		if profile_from_api.time_zone == "null":
			self.zone = "unknown"
		else:
			self.zone = profile_from_api.time_zone


	def get_new_features(self):
		# This is based on a Tweet database! So it must be created before!
		database = Database("twitter")
		table  = "tweets"
		retweets = database.select_data_for_author(self.author_id,"tweets","retweets")
		favorites = database.select_data_for_author(self.author_id,"tweets","favorites")
		database.close_connection()
		
		self.ntweets = len(retweets)
		self.sum_retweets = sum(retweets)
		self.median_retweets = median(retweets)
		self.max_retweets = max(retweets)

		self.sum_favorites = sum(favorites)
		self.median_favorites = median(favorites)
		self.max_favorites = max(favorites)

		self.retweets_per_tweet = float(self.sum_retweets)/float(self.ntweets)
		self.favorites_per_tweet = float(self.sum_favorites)/float(self.ntweets)
		#for key in self.__dict__.keys():
		#	print key,self.__dict__[key]

		

	def columns(self):
		return {
		"author_id":"BIGINT",
		"name":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
		"image_url":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
		"description":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
		"screen_name":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
		"topic":"TEXT",
		"ntweets":"INTEGER",
		"contributors":"INTEGER",
		"alltweets":"INTEGER",
		"sum_retweets":"INTEGER",
		"sum_favorites":"INTEGER",
		"followers":"INTEGER",
		"friends":"INTEGER",
		"max_retweets":"INTEGER",
		"max_favorites":"INTEGER",
		"median_retweets":"REAL",
		"median_favorites":"REAL",
		"retweets_per_tweet":"REAL",
		"favorites_per_tweet":"REAL",
		"fame":"REAL",
		"listed":"INTEGER",
		"zone":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
		}

####################################
# - TWEET -------------------------

class Tweet:
	def __init__(self,topic):
		self.topic = topic
		self.id = None
		self.author_id = None
		self.content = None
		self.retweets = None
		self.favorites = None
		self.rt = None
		self.question = None
		self.length = None
		self.nlinks = None
		self.nat = None
		self.nhash = None
		self.age = None
		self.date = None
		self.day = None
		self.month = None
		self.year = None
		self.hour = None
		self.words = None
		self.mean_word = None
		self.upper2lower = None
		self.digits = None
		self.popular = None
		self.loved = None
		self.language = None
		self.metric1 = 0.
		self.metric2 = 99.
		self.metric3 = 0.
		self.truncated = None
		self.source = None

	def columns(self):
		return []

	def isComplete(self):
		for variable in self.__dict__.keys():
			if self.__dict__[variable] == None:
				return False
		return True

	# Features of a tweet:
	# tweet_from_api is a tweet extracted by api not api itself
	def get_original_features(self,tweet_from_api):
		self.id = tweet_from_api.id
		self.author_id = tweet_from_api.user.id
		self.content = tweet_from_api.text
		self.retweets = tweet_from_api.retweet_count
		self.favorites = tweet_from_api.favorite_count
		self.date = tweet_from_api.created_at
		self.language = tweet_from_api.user.lang
		self.source = tweet_from_api.source

		if tweet_from_api.truncated:
			self.truncated = 1
		else:
			self.truncated = 0


	def isRT(self):
		if re.search(r'RT',self.content) == None:
			self.rt = 0
		else:
			self.rt = 1


	def get_retweets_ratio(self):
		self.retweets_ratio = float(self.retweets)/float(self.followers)

	def get_favorites_ratio(self):
		self.favorites_ratio = float(self.favorites)/float(self.followers)

	def isQuestion(self):
		if re.search(r'\?',self.content) == None:
			self.question = 0
		else:
			self.question = 1

	def get_length(self):
		self.length = len(self.content)

	def get_links(self):
		self.nlinks = len([s.start() for s in re.finditer('http', self.content)])

	def get_at(self):
		self.nat = len([s.start() for s in re.finditer('@', self.content)])

	def get_hash(self):
		self.nhash = len([s.start() for s in re.finditer('#', self.content)])

	def get_age(self):
		t0 = time.time()
		t1 = time.mktime(time.strptime(str(self.date),"%Y-%m-%d %H:%M:%S"))
		#t2 = time.mktime(time.strptime("2013-09-07 00:00:00","%Y-%m-%d %H:%M:%S"))
		#t3 = time.mktime(time.strptime("time.gmtime(),")
		self.age = (t0-t1)/3600./24.

	def get_day(self):
		dt = time.strptime(str(self.date),"%Y-%m-%d %H:%M:%S")
		self.day = dt.tm_wday
	
	def get_month(self):
		dt = time.strptime(str(self.date),"%Y-%m-%d %H:%M:%S")
		self.month = dt.tm_mon
	
	def get_year(self):
		dt = time.strptime(str(self.date),"%Y-%m-%d %H:%M:%S")
		self.year = dt.tm_year

	def get_hour(self):
		dt = time.strptime(str(self.date),"%Y-%m-%d %H:%M:%S")
		self.hour = dt.tm_hour
	
	def get_words(self):
		w = len(self.content.split())
		l = self.nlinks
		a = self.nat
		h = self.nhash
		self.words = w - l - a - h

	def get_mean_word(self):
		words = self.content.split()
		new_words = []
		for w in words:
			if not re.match(r'^#',w) and not re.match(r'^http',w) and not re.match(r'^@',w):
				new_words.append(len(w))
		# I am adding 1 below to account for tweets that have no words except for links etc	
		self.mean_word = sum(new_words)/(len(new_words)+1)

	def get_upper2lower_and_digts(self):
		words = self.content.split()
		upper = 0
		lower = 0
		digit = 0
		for w in words:
			if not re.match(r'^#',w) and not re.match(r'^http',w) and not re.match(r'^@',w) and not re.match(r'RT',w):
				lower = len([s.start() for s in re.finditer('[a-z]', w)])
				upper = len([s.start() for s in re.finditer('[A-Z]', w)])
				digit = len([s.start() for s in re.finditer('[0-9]', w)])
						
		# I am adding 1 below to account for tweets that have no words except for links etc
		self.upper2lower = float(upper)/(float(lower)+1)
		self.digits = digit

	def get_popular(self,threshold):
		if self.retweets > threshold:
			self.popular = 1
		else:
			self.popular = 0

	def get_loved(self,threshold):
		if self.favorites > threshold:
			self.loved = 1
		else:
			self.loved = 0


	def get_new_features(self):
		self.isRT()
		self.isQuestion()
		self.get_length()
		self.get_links()
		self.get_at()
		self.get_hash()
		self.get_age()
		self.get_day()
		self.get_month()
		self.get_year()
		self.get_hour()
		self.get_words()
		self.get_mean_word()
		self.get_upper2lower_and_digts()
		self.get_popular(3)
		self.get_loved(5)

	def columns(self):
		return {
		"topic": "TEXT",
		"id": "BIGINT PRIMARY KEY",
		"author_id":"BIGINT",
		"content":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
		"retweets":"INTEGER",
		"favorites":"INTEGER",
		"rt":"BOOLEAN",
		"question":"BOOLEAN",
		"length":"INTEGER",
		"nlinks":"INTEGER",
		"nat":"INTEGER",
		"nhash":"INTEGER",
		"age":"REAL",
		"date":"DATETIME",
		"day":"INTEGER",
		"month":"INTEGER",
		"year":"INTEGER",
		"hour":"INTEGER",
		"words":"INTEGER",
		"mean_word":"REAL",
		"upper2lower":"REAL",
		"digits":"INTEGER",
		"popular":"BOOLEAN",
		"loved":"BOOLEAN",
		"language":"TEXT",
		"metric1":"REAL",
		"metric2":"REAL",
		"metric3":"REAL",
		"truncated":"BOOLEAN",
		"source":"VARCHAR (255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
		}



	def random_forest_columns(self):
		# These are columns used in the random forest fitting
		return ["metric1",'words','retweets_per_tweet',"length","nlinks","nat",
			"nhash","mean_word","upper2lower","max_retweets",
			"followers","median_retweets","digits","fame","alltweets","question"]

	def logistic_regression_columns(self):
		# These are columns used in the random forest fitting
		return ["metric1",'words',"length","nlinks","nat",
			"nhash","mean_word","upper2lower","max_retweets",
			"followers","median_retweets","digits","day",
			"question","max_favorites",
			"median_favorites","sum_retweets","ntweets","alltweets","friends","fame"]

	def logistic_regression_columns_ds(self):
		# These are columns used in the random forest fitting
		return ["metric1",'words',"length","nlinks","nat",
			"nhash","mean_word","upper2lower","max_retweets",
			"followers","median_retweets","digits","day",
			"question","max_favorites","retweets_per_tweet",
			"median_favorites","sum_retweets","ntweets","alltweets","friends","fame"]

	def logistic_regression_columns_c(self):
		# These are columns used in the random forest fitting
		return ["metric1",'nat','retweets_per_tweet','fame','question']


	def logistic_regression_columns_s(self):
	#	# These are columns used in the random forest fitting
		return ["metric1",'nat','words','nlinks','length','question']



####################################
# - TWITTER API --------------------

class TwitterApi():
	def __init__(self):
		CONSUMER_KEY = "xY7gwn7nmixV7cCOtn8fQ"
		CONSUMER_SECRET = "hfzNG8kdfownp1FQvoLh2ahvIHK63Q8984KrVVb2k"
		OAUTH_TOKEN = "608622014-qnAFh9XKBBMGOJyKUUdgfH3AexK6uwYH2QKQ6Ka8"
		OAUTH_TOKEN_SECRET = "NDz9d31mfWKM14iNJutwvCv0MRO2EgG0SqrEwjVeo"
		auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
		auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
		self.api = tweepy.API(auth)
		

	def get_timeline_tweets(self,count):
		tweets = []
		return self.api.home_timeline(count=count)


	def get_timeline_for_friend(self,id,max_id,count):
		if max_id == None:
			return self.api.user_timeline(id=id,count=count)
		else:
			return self.api.user_timeline(id=id,count=count,max_id=max_id)
	
	def get_list_timeline(self,since_id,topic,count):
		if topic == "data science":
			lista = "Data"
		elif topic == "sport":
			lista = "Sport"
		elif topic == "celebrity":
			lista = "Celebrity"

		if since_id == None:
			return self.api.list_timeline('AnnaWanda1',lista, count=count)
		else:
			return self.api.list_timeline('AnnaWanda1',lista,count=count,since_id=since_id)



	def get_friend(self,id):
		return self.api.get_user(id)

	def get_list_authors(self,list):
		ids = []
		for x in tweepy.Cursor(self.api.list_members,'AnnaWanda1',list).pages(10):
			ids.extend([u.id for u in x])
		return ids
		#return [x.id for x in self.api.list_members('AnnaWanda1',list,page=2)]
		#for member in tweepy.Cursor(api.lists(), 'AnnaWanda1').items():
		#	print member

	def get_friends_ids(self,topic):
		if topic == "data science":
			return self.get_list_authors("Data")
		elif topic == "sport":
			return self.get_list_authors("Sport")
		elif topic == "celebrity":
			return self.get_list_authors("Celebrity")
		
if __name__ == "__main__":

	def get_hash(text):
		p = re.compile(r"(#[0-9a-zA-Z]+)")
		return p.sub(namedict["\1"],text)

	print get_hash("#ala ma kota #sdf02hsg")
	

