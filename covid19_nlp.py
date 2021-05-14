import os
import pandas as pd
from rake_nltk import Rake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# general research area: gauging people's reactions to COVID-19 over time
# based on a subset of tweets on Twitter
# analyze contents of tweets - are there repeated ideas, concepts, or people being mentioned?
# what are the key words coming from this data?
# how has public opinion changed over time? are there any meaningful changes in public sentiment?

# start with tweets from March 2020
# then check tweets in June 2020
# end by assessing tweets about the vaccine

os.chdir("C:\\Users\\ynkar\\Desktop\\computational_health\\tweets_kaggle_Mar2020")

# https://stackoverflow.com/questions/11361985/output-data-from-all-columns-in-a-dataframe-in-pandas
pd.set_option('display.max_columns', None)

# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

march_tweets_train = pd.read_csv("Corona_NLP_train_utf8.csv")
march_tweets_test = pd.read_csv("Corona_NLP_test_utf8.csv")
# print(march_tweets_test)

# https://www.kaggle.com/gpreda/covid19-tweets

os.chdir("C:\\Users\\ynkar\\Desktop\\computational_health\\tweets_kaggle_Jul2020")

july_tweets = pd.read_csv("covid19_tweets.csv")
# print(july_tweets)

# https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets

os.chdir("C:\\Users\\ynkar\\Desktop\\computational_health\\vaccination_tweets_kaggle")

vaccine_tweets = pd.read_csv("vaccination_all_tweets.csv")
# print(vaccine_tweets)

# take a subset of the csv and put it into a dataframe
# https://www.askpython.com/python/examples/subset-a-dataframe
march_df = march_tweets_train[["UserName", "TweetAt", "OriginalTweet", "Sentiment"]]
march_df.append(march_tweets_test[["UserName", "TweetAt", "OriginalTweet", "Sentiment"]])
print(march_df)

july_df = july_tweets[["date", "text", "hashtags"]]

# convert user_name to integer
# start with index 0, 1, ..., n
# add this to a list
# and add the list to the df as the "username" column

# https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
# https://stackoverflow.com/questions/18265935/python-create-list-with-numbers-between-2-values
id_list = list(range(1, len(july_df) + 1))
july_df.insert(0, "id", id_list, False)

print(july_df)

vaccine_df = vaccine_tweets[["id", "date", "text", "hashtags"]]

# create method to obtain key words using rake-nltk
# also perform sentiment analysis on each tweet using vader
# https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/

# we can get key words with or without scores
# https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f

def get_key_words(df: pd.DataFrame, length: int, tweet: str, id, time: str, score_boolean: bool):
    r = Rake()
    sid = SentimentIntensityAnalyzer()

    # store the user number as key and top 3 words by rake-nltk in a dictionary (hashTable)
    keyword_dict = {}

    for i in range(length):
        r.extract_keywords_from_text(df[tweet][i])

        if score_boolean:
            provided_words = r.get_ranked_phrases_with_scores()
        else:
            provided_words = r.get_ranked_phrases()

        # store the top three results of each user in a hashTable
        # this provides quick lookup and convenient storage of data
        # also store date of tweet

        keyword_dict[df[id][i]] = provided_words[0:3]
        keyword_dict[df[id][i]].append(df[time][i])

        sentiment_dict = sid.polarity_scores(df[tweet][i])

        sentiment = ""
        if sentiment_dict['compound'] > 0.05:
            sentiment = "Positive"
        elif sentiment_dict['compound'] < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # add vader sentiment to the list of tuples in keyword dictionary
        keyword_dict[df[id][i]].append(sentiment)

    print(keyword_dict)

get_key_words(march_df, 20, "OriginalTweet", "UserName", "TweetAt", True)
get_key_words(july_df, 10, "text", "id", "date", True)
get_key_words(vaccine_df, 10, "text", "id", "date", True)

print("")

get_key_words(march_df, 10, "OriginalTweet", "UserName", "TweetAt", False)
get_key_words(july_df, 10, "text", "id", "date", False)
get_key_words(vaccine_df, 10, "text", "id", "date", False)