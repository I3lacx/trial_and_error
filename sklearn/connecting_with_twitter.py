import tweepy
from textblob import TextBlob

import csv
consumer_key = "gw1gDdmdf76PQzK25HA8FnPZB"
consumer_secret = "QPWk2gkkP8Xb1Ck03fLeDY5CFOv2dXMNRG2hNN9BGWRyyjDF5r"

acces_token = "322031988-M4TgCrzJqUKmGBC4unBg5nS2yb8PcRtwCiRNPgfW"
acces_token_secret = "qkCsMqdhlyUF3oCr6TpeQCIQQjiNQiBZq06XhCcEyyNIK"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(acces_token, acces_token_secret)

api = tweepy.API(auth)
#api.update_status(status = 'This is robot')

public_tweets = api.search('Trump')



path = "E:/Programmieren/Python/learnPython/trial_and_error/sklearn/"
ptf = path + "twitterPosts.csv"

#w+ for write and create if non existent
with open(ptf, 'w+') as csv_file:
    label_names = ["tweet", 'label']
    writer = csv.DictWriter(csv_file, fieldnames=label_names)
    writer.writeheader()
    label = "unknown"
    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        positivity = analysis.sentiment.polarity
        if(positivity > 0):
            label = "positive"
        else:
            label = "negative"

        tweet_text = tweet.text.encode('utf-8')
        writer.writerow({'tweet' : tweet_text, 'label' : label})
