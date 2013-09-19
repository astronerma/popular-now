import MySQLdb
import tweepy
import os.path
from my_classes import Author,Tweet,Database,TwitterApi

# Each topic is related to different database but also to different api keys!
# See in twitter_api.py which api to choose!



def download_tweets(table,topic,count,page):
	print "Downloading tweets for: "+ topic
	# Prepare database for your tweets
	database = Database("twitter")
	# Connect to twitter API
	api = TwitterApi()
	# Get list of friends from the database! not from api!
	friends_ids = api.get_friends_ids(topic)
	# Check if tweets table exists
	# If yes drop it only if downloading first page
	# Get timeline for each friend
	for id in friends_ids:
		if page == 1:
			min_tweet = None
			tweets = api.get_timeline_for_friend(id,None,count)
		else:
			min_tweet = database.get_min_id(id,table,topic)
			tweets = api.get_timeline_for_friend(id,min_tweet-1,count)
		
		#print id,min_tweet
		
		for t in tweets:
			tweet = Tweet(topic)
			tweet.get_original_features(t)
			tweet.get_new_features()
			database.insert(table,tweet)
		
	database.close_connection()





def create_author_database():
	print "Creating table authors"
	database = Database("twitter")
	database.create_table("authors",Author("data science").columns())
	database.close_connection()


def create_tweets_table(table):
	print "Creating table",table
	database = Database("twitter")
	database.create_table(table,Tweet("data science").columns())
	database.close_connection()



def download_authors(topic):
	print "Downloading authors for: "+ topic
	# Prepare database for your authors
	database = Database("twitter")
	table = "authors"

	# Connect to twitter API
	api = TwitterApi()

	# Get list of friends for topic
	# We are only getting friends for our topic
	# What would happen if there was the same author for both topics?
	# I guess it doesn't matter
	friends_ids = api.get_friends_ids(topic)

	# Insert each friend into database
	for id in friends_ids:
		f = api.get_friend(id)
		author = Author(topic)
		author.get_original_features(f)
		author.get_new_features()
		
		#database.insert(table,author)
		database.update_author("authors",author,author.author_id)


	database.close_connection()


def get_latest_tweets(topic,count):
	# Prepare database for your tweets
	database = Database("twitter")
	# Connect to twitter API
	api = TwitterApi()
	
	max_tweet = database.get_max_id("new")
	if max_tweet != None:
		max_tweet += 1
	tweets = api.get_list_timeline(max_tweet,topic,count)
	print "found new tweets: ",len(tweets)

	for t in tweets:
		tweet = Tweet(topic)
		tweet.get_original_features(t)
		tweet.get_new_features()
		#print "inserting"
		database.insert("new",tweet)
		
	database.close_connection()

def delete_too_old_tweets(topis,age):
	pass


if __name__ == "__main__":
	# Instruction
	# If starting from scratch empty database
	# First create tables
	# Then download initial data
	# Then download more data
	# Once you have many tweets download author data
	# Before running the web page make sure you have your model file in place 
	# If not, run random forest file it will dump a classifier to a file
	# The website should work now
	# If not check if 'now' table is filled properly

	# Create tables
	#create_tweets_table("tweets")
	#create_tweets_table("new")
	#create_author_database()

	# Fill the first batch of data
	#download_tweets("tweets","data science",200,1)
	#download_tweets("tweets","sport",200,1)
	#download_tweets("tweets","celebrity",200,1)

	# Download more data
	#download_tweets("tweets","data science",200,2)
	#download_tweets("tweets","sport",200,2)
	#download_tweets("tweets","celebrity",200,2)


	# Run these only at the end, once you have all tweets
	download_authors("data science")
	download_authors("sport")
	download_authors("celebrity")
	
	# !!!!!!!!!!!!!!!!!!!
	# If tweets don't refresh, run this!!!!!!


	# Before running any models run this!
	# At some point you may want to update the database metric value
	# run this every time you download any data to tweets database!
	#database = Database("twitter")
	#database.create_new_metric()
	#database.close_connection()


	# After that train both NB and RF!!!! for all three topics


	# Before first use of web app use this
	#create_table("new")
	#get_latest_tweets("data science",200)
	#get_latest_tweets("sport",200)
	#get_latest_tweets("celebrity",200)








