import os
import pandas as pd
import twint

# Create table from twitter
def tweet_create_table():
    # Create a search list
    search_list = {
        "#distancelearning": 10000,
        "#onlineschool": 10000,
        "#onlineteaching": 10000,
        "#virtuallearning": 10000,
        "#onlineducation": 10000,
        "#distanceeducation": 10000,
        "#OnlineClasses": 10000,
        "#DigitalLearning": 10000,
        "#elearning": 10000,
        "#onlinelearning": 10000,
    }

    pd.options.mode.chained_assignment = None

    # Configure twint
    c = twint.Config()
    c.Pandas = True
    c.Since = "2020-04-17"
    c.Lowercase = True
    c.Hide_output = True

    # Create dataframe directories
    if not os.path.exists("./tweet_en_dataframes/"):
        os.mkdir("./tweet_en_dataframes/")

    # Create full dataframe
    df_full = pd.DataFrame()

    # Seach tweet hash tags or serach words
    for word, limit in search_list.items():
        c.Search = word
        c.Limit = limit
        print("Twint keyword search " + word + " is started...")
        twint.run.Search(c)
        print("Twint keyword search " + word + " is completed...")

        # Save in pandas dataframe filter out foreign tweets, drop duplicate tweets and drop NaN tokens
        df = twint.storage.panda.Tweets_df
        df = df[df["language"] == "en"]
        df = df.drop_duplicates(subset=["tweet"])

        # Feature Selection total 19 columns removed
        # In case of some tweets could broke csv file by adding enter character so remove this column
        del df["search"]
        del df["link"]
        del df["hour"]
        del df["day"]
        del df["translate"]
        del df["trans_src"]
        del df["trans_dest"]
        del df["geo"]
        del df["place"]
        del df["timezone"]
        # Since we have only one filtered
        del df["language"]
        # Erase other empty values
        del df["near"]
        del df["source"]
        del df["user_rt_id"]
        del df["user_rt"]
        del df["retweet_id"]
        del df["retweet_date"]
        del df["quote_url"]
        del df["thumbnail"]

        df.to_csv("./tweet_en_dataframes/tweet_en_" + word + ".csv")
        print(word + " is completed...")
        df_full = pd.concat([df_full, df], axis=0)
        df_full.to_csv("tweet_en_full_test.csv")


if __name__ == "__main__":
    tweet_create_table()
