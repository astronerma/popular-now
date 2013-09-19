import random
import re
import tweepy
import pickle
import MySQLdb
import string
import time
from app import app
from flask import render_template, request, flash, redirect, url_for 
from decimal import Decimal
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from my_classes import Author,Tweet,Database,TwitterApi

database = Database("twitter")
namedict = database.get_names()


@app.route('/')

@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/authors')
def authors():

	database = Database("twitter")
	a = database.get_author_info("data science")
	#print authors
	authors1 = []
	for row in a:
		img = row[1]
		img = re.sub('normal','bigger',img)	
		authors1.append({
			"name": row[0],
			"image": img,
			"description": row[2],
			"followers": row[3],
			"retweets_per_tweet": "{:10.1f}".format(row[4]),
			"screen_name":row[5]
			})

	database = Database("twitter")
	a = database.get_author_info("celebrity")
	#print authors
	authors2 = []
	for row in a:
		img = row[1]
		img = re.sub('normal','bigger',img)	
		authors2.append({
			"name": row[0],
			"image": img,
			"description": row[2],
			"followers": row[3],
			"retweets_per_tweet": "{:10.1f}".format(row[4]),
			"screen_name":row[5]
			})

	database = Database("twitter")
	a = database.get_author_info("sport")
	#print authors
	authors3 = []
	for row in a:
		img = row[1]
		img = re.sub('normal','bigger',img)	
		authors3.append({
			"name": row[0],
			"image": img,
			"description": row[2],
			"followers": row[3],
			"retweets_per_tweet": "{:10.1f}".format(row[4]),
			"screen_name":row[5]
			})

	database.close_connection()
	return render_template("authors.html",aD=authors1,aC=authors2,aS=authors3)


# http://opentechschool.github.io/python-flask/core/form-submission.html
@app.route('/new_form')
def new_form():
	return render_template("new_form.html")

@app.route('/result_form', methods = ['POST'])
def result_form():
	username = request.form['username']
	friends = get_friends()
	user = get_user_info(username)
	return render_template("result_form.html",user=user)

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

def calculate_class(classifierRF,classifierNB,row,columns):
	features = []
	for c in columns:
		features.append(row[c])
		
	myclassRF = classifierRF.predict(features)[0]
	myclassNB = classifierNB.classify(prepare_content(row['content']))
		
	if myclassRF == 1.0 and myclassRF == myclassNB:
		myclass = 1.0
	else:
		myclass = 0.0

	if row["retweets"] > row["metric2"]+1:
		myclass = 2.0

	return myclass

def author_name(match):
	global namedict
	match = match.group()
	#print match
	return namedict[match]


@app.route('/feed',methods = ['GET'])
# This function will display my new feed!
# Here I will also filter tweets!
def feed():
	topic = request.args['topic']
	#print topic
	database = Database("twitter")
	database.change_cursor()

	# change content
	phash = re.compile(r"(#[0-9a-zA-Z]+)")
	phttp = re.compile(r"(http://t\.co/[0-9a-zA-Z/:]+)")
	phttps = re.compile(r"(https://t\.co/[0-9a-zA-Z/:]+)")
	#pat = re.compile(r"@([0-9a-zA-Z]+)")


	# Download data from database
	q = "SELECT * FROM new,authors WHERE new.rt = 0 and new.author_id = authors.author_id and \
		new.topic= " + "\"" + topic + "\"" + " ORDER BY date DESC;"
	#print q
	all_tweets = database.query(q)
	#print q

	# time now
	t0 = time.time()
	timezone_offset = 7.*3600.
	
	tweets = []
	for row in all_tweets:	
		# time of the tweet
		t1 = time.mktime(time.strptime(str(row['date']),"%Y-%m-%d %H:%M:%S"))
		
		age = (t0-t1+timezone_offset)/3600.
		if age < 1.:
			age = str(int(age*60))+" min"
		elif age >= 1. and age < 24.:
			age = str(int(age))+" hrs"
		else:
			age = str(int(age/24))+" d"

		
		if row['metric2'] == 1:
			# ----------------------
			text = row['content']
			text = phttps.sub("<i class=\"icon-hand-right\"></i><a href=\"\\1\" target=\"_blank\"> link </a>",text)
			text = phttp.sub("<i class=\"icon-hand-right\"></i><a href=\"\\1\" target=\"_blank\"> link </a>",text)
			#text = phash.sub("<span title=\"\\1\"><i class=\"icon-tag\"></i></span>",text)
			
			text = phash.sub("<i class=\"icon-tag tooltips\" data-toggle=\"tooltip\" \
				data-html=\"true\" title=\"<h4>\\1</h4>\"></i>",text)



			text = "<td><blockquote class=\"twitter-tweet\"><p>"+text+"</p>&mdash; " + row['name'] + " (@"+row['screen_name']+") </blockquote></td>"
			image = "<td width=\"50\"><img src="+ row['image_url'] +" class=\"img-circle\"></td>"
			#age = "<td width=\"50\"><p>"+ age +"</p></td>"
			age = "<td width=\"50\"><p>"+ age + "<h4><i class=\"tooltips icon-asterisk badge-color2\" data-toggle=\"tooltip\" \
			data-html=\"true\" title=\"<h4>Prediction</h4>\" data-placement=\"right\"></i></h4></p></td>"


		
		elif row['metric2'] == 2:
			# ----------------------
			text = row['content']
			text = phttps.sub("<i class=\"icon-hand-right\"></i><a href=\"\\1\" target=\"_blank\"> link </a></i>",text)
			text = phttp.sub("<i class=\"icon-hand-right\"></i><a href=\"\\1\" target=\"_blank\"> link </a></i>",text)
			#text = phash.sub("<span title=\"\\1\"><i class=\"icon-tag\"></i></span>",text)
			text = phash.sub("<i class=\"icon-tag tooltips\" data-toggle=\"tooltip\" \
				data-html=\"true\" title=\"<h4>\\1</h4>\"></i>",text)

			text = "<td><blockquote class=\"twitter-tweet\"><p>"+text+"</p>&mdash; " + row['name'] + " (@"+row['screen_name']+") </blockquote></td>"
			image = "<td width=\"50\"><img src="+ row['image_url'] +" class=\"img-circle\"></td>"
			#age = "<td width=\"50\"><p>"+ age +"<h4><i class=\"icon-asterisk badge-color\"></i></h4></p></td>"
			age = "<td width=\"50\"><p>"+ age + "<h4><i class=\"tooltips icon-asterisk badge-color\" data-toggle=\"tooltip\" \
			data-html=\"true\" title=\"<h4>Already popular</h4>\" data-placement=\"right\"></i></h4></p></td>"
						

		else:
			# ----------------------
			text = row['content']
			text = phttps.sub("<i class=\"icon-hand-right\"> <a class=\"inactive-link\" href=\"\\1\" target=\"_blank\">  link </a></i>",text)
			text = phttp.sub("<i class=\"icon-hand-right\"> <a class=\"inactive-link\" href=\"\\1\" target=\"_blank\">  link </a></i>",text)
			#text = phash.sub("<span title=\"\\1\"><i class=\"icon-tag\"></i></span>",text)
			text = phash.sub("<i class=\"icon-tag tooltips inactive-link\" data-toggle=\"tooltip\" \
				data-html=\"true\" title=\"<h4>\\1</h4>\"></i>",text)

			text = "<del>" + text
			text = "<td><blockquote class=\"twitter-tweet\" style=\"color: #C0C0C0\"><p>"+text+"</p> " + row['name'] + " (@"+row['screen_name']+") </blockquote></td>"
			image = "<td width=\"50\"><h1><i class=\"icon-twitter twitter-color\"></i></h1></td>"
			age = "<td width=\"50\"><p class=\"twitter-color\">"+ age +"</p></td>"

		if row['rt'] == 1.:
			row['metric2'] == 1.

		tweets.append({
			"content": text,
			"name": row["name"],
			"retweets": row["retweets"],
			"screen_name":row["screen_name"],
			"id":row["id"],
			"age":age,
			"myclass":row["metric2"],
			"image":image,
			
			})
	
	#print tweets
	if topic == "data science":
		name = "Data science and data visualization"
	elif topic == "celebrity":
		name = "Celebrity gossips"
	elif topic == "sport":
		name = "Sports news"
	
	database.close_connection()
	return render_template("feed.html",tweets=tweets,topic=topic,rt=0,name=name)



@app.route('/refresh_feed',methods=['POST'])
def refresh_feed():
	topic = request.form['topic']
	#print "i am here",topic
	database = Database("twitter")
	#database.show()
	database.get_latest_tweets(topic,100)
	database.delete_older_tweets(24)
	url = "/feed?topic=" + topic

	# columns used in my model
	#columns = Tweet("data science").random_forest_columns()[1:]
	columns = Tweet("data science").logistic_regression_columns()[1:]

	# update predicted class here
	# Only for tweets that have metric2 = 0
	database.update_class_in_new_table(columns,topic)
	# update metric in new table
	#database.update_all_tweets_in_new_table()
	database.close_connection()
	



	return redirect(url)
	#return render_template("refresh_feed.html",topic=topic)


# -------------------------


@app.route('/feedRT',methods=["GET"])
def feedRT():
	topic = request.args['topic']
	#print topic
	database = Database("twitter")
	database.change_cursor()
	q = "SELECT * FROM new,authors WHERE new.rt = 1 and new.author_id = authors.author_id and \
		new.topic= " + "\"" + topic + "\"" + " ORDER BY date DESC;"

	# change content
	phash = re.compile(r"(#[0-9a-zA-Z]+)")
	phtml = re.compile(r"(http://t\.co/[0-9a-zA-Z/:]+)")
	pat = re.compile(r"(@[0-9a-zA-Z]+)")

	# time now
	t0 = time.time()
	timezone_offset = 7.*3600.



	all_tweets = database.query(q)
	#print q
	tweets = []
	for row in all_tweets:		
	
		# time of the tweet
		t1 = time.mktime(time.strptime(str(row['date']),"%Y-%m-%d %H:%M:%S"))
		
		age = (t0-t1+timezone_offset)/3600.
		if age < 1.:
			age = str(int(age*60))+" min"
		elif age >= 1. and age < 24.:
			age = str(int(age))+" hrs"
		else:
			age = str(int(age/24))+" d"

		
		text = row['content']
		text = phtml.sub("<i class=\"icon-hand-right\"></i><a href=\"\\1\" target=\"_blank\"> link </a>",text)
		
		text = phash.sub("<span title=\"\\1\"><i class=\"icon-tag\"></i></span>",text)


		#text = phash.sub("<a href=\"#\" data-toggle=\"tooltip\" class=\"#hashtag\"  title=\"\\1\"><i class=\"icon-tag\"></i></a>",text)
		
		

		text = "<td><blockquote class=\"twitter-tweet\"><p>"+text+"</p>&mdash; " + row['name'] + " (@"+row['screen_name']+") </blockquote></td>"
		image = "<td width=\"50\"><img src="+ row['image_url'] +" class=\"img-circle\"></td>"
		age = "<td width=\"50\"><p>"+ age +"</p></td>"

	

		tweets.append({
			"content": text,
			"name": row["name"],
			"retweets": row["retweets"],
			"screen_name":row["screen_name"],
			"id":row["id"],
			"age":age,
			"myclass":row["metric2"],
			"image":image,
			
			})
	
	#print tweets
	if topic == "data science":
		name = "Data science and data visualization"
	elif topic == "celebrity":
		name = "Celebrity gossips"
	elif topic == "sport":
		name = "Sports news"
	
	database.close_connection()
	return render_template("feed.html",tweets=tweets,topic=topic,rt=1,name=name)

@app.route('/model',methods=["GET"])
def model():
	return render_template("model.html")


@app.route('/slides')
def slides():
	return render_template("slides.html")




