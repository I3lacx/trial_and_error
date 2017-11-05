import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense


#Step 1 - Insert your API keys
consumer_key = "gw1gDdmdf76PQzK25HA8FnPZB"
consumer_secret = "QPWk2gkkP8Xb1Ck03fLeDY5CFOv2dXMNRG2hNN9BGWRyyjDF5r"
acces_token = "322031988-M4TgCrzJqUKmGBC4unBg5nS2yb8PcRtwCiRNPgfW"
acces_token_secret = "qkCsMqdhlyUF3oCr6TpeQCIQQjiNQiBZq06XhCcEyyNIK"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Search for your company name on Twitter
public_tweets = api.search('Google')


#Step 3 - Define a threshold for each sentiment to classify each
#as positive or negative. If the majority of tweets you've collected are positive
#then use your neural network to predict a future price
for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)


#data collection
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

#Step 5 reference your CSV file here
get_data('E:/Programmieren/Python/learnPython/trial_and_error/sklearn/googl2.csv')

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price
#on a given day. We'll later print the price out to terminal.
def predict_prices(dates, prices, x):
    model = Sequential()
    model.add(Activation('tanh'))

    model.fit(dates, prices, epochs=5, batch_size=32)
    return model.predict(dates, prices)

predicted_price = predict_price(dates, prices, 29)
print(predicted_price)
