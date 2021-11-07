import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import json

class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''
		# keys and tokens from the Twitter Dev Console
		consumer_key = 'iGXFMDC2ZGxRAYRuGBeXiFVRp'
		consumer_secret = 'Q4ds00pfGrSuqm7yW8iNbGSd62F8eujbQvUqIyNpJHLhR2SXim'
		access_token = '430675318-TkwHiIeF5ykgURfMutpuU6mgmVYXMruYhLgQrVnL'
		access_token_secret = 'e4u5CrskPiikU8Hn1rKEcw1ljfPOjxkMBVjXPFzopxeZd'

		# attempt authentication
		try:
			# create OAuthHandler object
			self.auth = OAuthHandler(consumer_key, consumer_secret)
			# set access token and secret
			self.auth.set_access_token(access_token, access_token_secret)
			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")

	def clean_tweet(self, tweet):
		'''
		Utility function to clean tweet text by removing links, special characters
		using simple regex statements.
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", tweet).split())

	def get_tweet_sentiment(self, tweet):
		'''
		Utility function to classify sentiment of passed tweet
		using textblob's sentiment method
		'''
		# create TextBlob object of passed tweet text
		analysis = TextBlob(self.clean_tweet(tweet))
		# set sentiment
		if analysis.sentiment.polarity > 0:
			return 'positive'
		elif analysis.sentiment.polarity == 0:
			return 'neutral'
		else:
			return 'negative'

	def get_tweets(self, query, result_type, count):
		'''
		Main function to fetch tweets and parse them.
		'''
		# empty list to store parsed tweets
		tweets = []

		try:
			# call twitter api to fetch tweets
			fetched_tweets = self.api.search(q = query, result_type = result_type, count = count)

			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}

				# saving text of tweet
				temp = re.sub('http://\S+|https://\S+', '', tweet.text)
				# temp = re.sub('(@[A-Za-z0-9]+)','',temp)
				parsed_tweet['text'] = temp
				# saving sentiment of tweet
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(temp)

				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)

			# return parsed tweets
			return tweets

		except tweepy.TweepError as e:
			# print error (if any)
			print("Error : " + str(e))

	def run(self, query_string):
		# calling function to get tweets
		tweets = self.get_tweets(query_string, 'mixed', 15)

		# picking positive tweets from tweets
		ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
		ptweetsPer = "Positive tweets percentage: {} %".format(round(100*len(ptweets)/len(tweets),2))
		print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))

		# picking negative tweets from tweets
		ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
		ntweetsPer = "Negative tweets percentage: {} %".format(round(100*len(ntweets)/len(tweets),2))
		print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))

		# percentage of neutral tweets
		netweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral']
		netweetsPer = "Neutral tweets percentage: {} % ".format(round(100 * len(netweets) / len(tweets),2))
		print("Neutral tweets percentage: {} % ".format(100*len(netweets)/len(tweets)))

		# # printing first 5 positive tweets
		# print("\n\nPositive tweets:")
		# for tweet in ptweets[:5]:
		# 	print(tweet['text'])
		# 	print("1111")
		#
		# # printing first 5 negative tweets
		# print("\n\nNegative tweets:")
		# for tweet in ntweets[:5]:
		# 	print(tweet['text'])

		return ptweetsPer, ntweetsPer, netweetsPer, ptweets, ntweets, netweets
#
#
#
# #
# #
# #
# query_string = 'Samsung'
# main(query_string)