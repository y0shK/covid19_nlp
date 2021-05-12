import os
import pandas as pd
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

tweets_train = pd.read_csv("Corona_NLP_train_utf8.csv")
tweets_test = pd.read_csv("Corona_NLP_test_utf8.csv")
print(tweets_test)

# https://www.kaggle.com/gpreda/covid19-tweets

# https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets