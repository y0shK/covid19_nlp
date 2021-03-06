import os
import pandas as pd
from rake_nltk import Rake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
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

# return None - this method only prints
def get_key_words(df: pd.DataFrame, length: int, tweet: str, id, time: str, score_boolean: bool) -> None:
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

        # make sure that none of the top three results are or have "https ://" or ". co"

        # score_boolean = false
        if not score_boolean:
            top_three = []
            for j in range(len(provided_words)):
                # case where the result is just the string to be removed
                if provided_words[j] == "https ://" or provided_words[j] == ". co":
                    pass
                # case where the result needs to remove the substring
                elif "https ://" in provided_words[j]:
                    index_to_split = provided_words[j].find("https ://")
                    top_three.append(provided_words[j][0:index_to_split - 1])
                elif ". co" in provided_words[j]:
                    index_to_split = provided_words[j].find(". co")
                    top_three.append(provided_words[j][0:index_to_split - 1])
                else:
                    top_three.append(provided_words[j])

                # if we have the top three words, we can break
                if len(top_three) >= 3:
                    break
        else:
            # this is the case where score_boolean is true and we just want the string of the tuple
            # which is tuple[1], while the score is tuple[0]
            top_three = []
            for j in range(len(provided_words)):
                # case where the result is just the string to be removed
                if provided_words[j][1] == "https ://" or provided_words[j][1] == ". co":
                    pass
                # case where the result needs to remove the substring
                elif "https ://" in provided_words[j][1]:
                    index_to_split = provided_words[j][1].find("https ://")
                    top_three.append(provided_words[j][1][0:index_to_split - 1])
                elif ". co" in provided_words[j][1]:
                    index_to_split = provided_words[j][1].find(". co")
                    top_three.append(provided_words[j][1][0:index_to_split - 1])
                else:
                    top_three.append(provided_words[j])

                # if we have the top three words, we can break
                if len(top_three) >= 3:
                    break

        # we now have the top three words for each unidentified user
        keyword_dict[df[id][i]] = top_three

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

#get_key_words(march_df, 20, "OriginalTweet", "UserName", "TweetAt", True)
#get_key_words(july_df, 10, "text", "id", "date", True)
#get_key_words(vaccine_df, 10, "text", "id", "date", True)

print(" ")

#get_key_words(march_df, 10, "OriginalTweet", "UserName", "TweetAt", False)
#get_key_words(july_df, 10, "text", "id", "date", False)
#get_key_words(vaccine_df, 10, "text", "id", "date", False)

# get frequency distribution using nltk
# create a method that generalizes and prints the frequency distribution
# https://stackoverflow.com/questions/46786211/counting-the-frequency-of-words-in-a-pandas-data-frame

# return None - this method only prints
def get_word_freq(df: pd.DataFrame, length: int, tweet: str, common_filter_bool: bool) -> None:
    col = df[tweet].str.lower().str.cat(sep=" ")
    words = nltk.tokenize.word_tokenize(col)
    word_dist = nltk.FreqDist(words)
    # print(word_dist)

    result = pd.DataFrame(word_dist.most_common(length),
                          columns=["Word", "Frequency"])
    # print(result)

    word_dict = {}

    # create a common word filter to eliminate articles, conjunctions, prepositions, etc.
    # and words like "https"

    common_filter = []
    articles = ["a", "an", "the"]
    conjunctions = ["for", "and", "nor", "but", "or", "yet", "so"]
    common_verbs = ["is", "are"]
    common_nouns = ["this", "that"]
    common_phrases = ["i", "you", "they", "their", "our", "we", "us"]
    other_words = ["https", ".co"]

    # https://stackoverflow.com/questions/328059/create-a-list-that-contain-each-line-of-a-file
    os.chdir("C:\\Users\\ynkar\\Desktop\\computational_health")
    prepositions = open("prepositions.txt").read().splitlines()

    common_filter += articles
    common_filter += conjunctions
    common_filter += prepositions
    common_filter += common_verbs
    common_filter += common_nouns
    common_filter += common_phrases
    common_filter += other_words

    # https://python-reference.readthedocs.io/en/latest/docs/str/isalnum.html
    # https://stackoverflow.com/questions/15125343/how-to-iterate-through-two-pandas-columns
    for word, freq in zip(result["Word"], result["Frequency"]):

        if common_filter_bool:
            if word.isalnum() and word not in common_filter:
                word_dict[word] = freq
        else:
            if word.isalnum():
                word_dict[word] = freq

    print(word_dict)

get_word_freq(march_df, 100, "OriginalTweet", False)
get_word_freq(july_df, 100, "text", False)
get_word_freq(vaccine_df, 100, "text", False)

get_word_freq(march_df, 100, "OriginalTweet", True)
get_word_freq(july_df, 100, "text", True)
get_word_freq(vaccine_df, 100, "text", True)