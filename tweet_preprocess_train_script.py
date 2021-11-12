# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
from textblob.blob import TextBlob
from nltk.corpus import stopwords
from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS
import json
import enchant


# %%
df = pd.read_csv("./tweet_en_full_test.csv")
abbrevations = open("tweet_abbrevations.json", "r")
abbrevations_lookup = json.load(abbrevations)

# %%
# Remove Uni characters
df["tweet"] = df["tweet"].str.encode("ascii", "ignore")
df["tweet"] = df["tweet"].str.decode("utf-8")


# %%
# Define a function to classify polarities
def tweet_create_label_polarity(polarity):
    if polarity > 0:
        return 2
    elif polarity < 0:
        return 0
    else:
        return 1

# %%
# Add stop words to nltk corpus just in case
stopwords = nltk.corpus.stopwords.words("english")
newStopWords = ["cant"]
stopwords.extend(newStopWords)
# Unfound words in preprocess
unfoundWord = []


# %%
# Preprocessing step
def tweet_preprocess(tweet):
    # Remove links
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # Remove mentions and hashtag, digits and ampersand character
    tweet = re.sub(r"(\@\w+|\#\w+|\&amp|[0-9]+|\_+)", "", tweet)

    # Remove punctionations ands replace with space
    tweet = re.sub(r"[^\w\s]", " ", tweet)

    # Lower words
    tweet = tweet.lower()

    # Tokenize the words
    tokenized = word_tokenize(tweet)

    # Remove the stop words and punctionation
    tokenized = [token for token in tokenized if token not in stopwords]

    return tokenized


def tweet_find_unfound(tokenized):
    # Lemmatize the words and abberation check
    lemmatizer = WordNetLemmatizer()
    unfoundToken = []
    phrase_check = re.compile("[_]")

    for index, token in enumerate(tokenized):
        # Do not take bigrams
        if phrase_check.search(token) == None:
            a = lemmatizer.lemmatize(token, pos="a")
            n = lemmatizer.lemmatize(token, pos="n")
            v = lemmatizer.lemmatize(token, pos="v")

            # Token may be adjective verb or noun
            if token != a and token != n and token != v:
                tokenized[index] = a
                continue
            elif token != a and token != n and token == v:
                tokenized[index] = n
                continue
            elif token != a and token == n and token != v:
                tokenized[index] = a
                continue
            elif token != a and token == n and token == v:
                tokenized[index] = a
                continue
            elif token == a and token != n and token != v:
                tokenized[index] = n
                continue
            elif token == a and token != n and token == v:
                tokenized[index] = n
                continue
            elif token == a and token == n and token != v:
                tokenized[index] = v
                continue

            # Spell check if all words are same and not lemmatized
            if d.check(token):
                continue
            # if spell is incorrect
            else:
                # Convert if it is a internet abbrevation (if any) to long ones (in future)
                isAbrevationFound = False
                # lookup abbrevations table
                for abrevation, value in enumerate(abbrevations_lookup):
                    if token == value["word"].lower() or token == value["word"].upper():
                        tokenized[index] = value["meaning"]
                        isAbrevationFound = True
                        break
                # if not found anything remove the word
                if isAbrevationFound == False:
                    unfoundWord.append(token)
                    unfoundToken.append(token)
            
        else:
            continue

    # Remove tokens that not founded
    for unfound in unfoundToken:
        tokenized.remove(unfound)

    # In case of some short tokens remain delete them
    tokenized = [token for token in tokenized if len(token) > 2]

    return tokenized


# %%
d = enchant.Dict("en_US")
tokens = df["tweet"].apply(tweet_preprocess)

# Use phraser before eliminating unfounded words
# phrases = Phrases(tokens, connector_words=ENGLISH_CONNECTOR_WORDS)
# bigram = Phraser(phrases)

# for index, token in tokens.items():
#     tokens[index] = bigram[tokens[index]]

# Find Unfound tokens and phrasal verbs
tokens = tokens.apply(tweet_find_unfound)


# %%
# Create series of words that not founded in dict and abbrevations
unfoundWordSeries = pd.Series(unfoundWord)
unfoundWordSeries = unfoundWordSeries.drop_duplicates()
# unfoundWordSeries = unfoundWordSeries.sort_values(ascending=True)
unfoundWordSeries.to_csv("tweet_unfoundWord.csv")


# %%
# Add polarity and subjectivity scores
df["tweet_polarity"] = np.nan
df["tweet_subjectivity"] = np.nan
for index, value in tokens.str.join(" ").items():
    b = TextBlob(value.replace("_", " "))
    df.at[index, "tweet_polarity"] = b.sentiment.polarity
    df.at[index, "tweet_subjectivity"] = b.sentiment.subjectivity

# Apply the funtion on Polarity column and add the results into a new column
df["tweet_polarity_label"] = df["tweet_polarity"].apply(tweet_create_label_polarity)
# Change the datatype as "category"
df["tweet_polarity_label"] = df["tweet_polarity_label"].astype("category")

# Balance values in case of repeating and this could cause overfit (could be used in future)
# count_class_positive, count_class_neutral, count_class_negative = df[
#     "tweet_polarity_label"
# ].value_counts()

# df_class_positive = df[df["tweet_polarity_label"] == 2]
# df_class_neutral = df[df["tweet_polarity_label"] == 1]
# df_class_negative = df[df["tweet_polarity_label"] == 0]

# df_class_positive_under = df_class_positive.sample(count_class_negative)
# df_class_neutral_under = df_class_neutral.sample(count_class_negative)
# df = pd.concat(
#     [df_class_positive_under, df_class_neutral_under, df_class_negative], axis=0
# )

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

graph_sentiment = df[['tweet_polarity', 'tweet_subjectivity', 'tweet_polarity_label']]


# Visualize the Polarity scores
plt.figure(figsize=(16,9)) 
sns.scatterplot(x="tweet_polarity", y="tweet_subjectivity", hue="tweet_polarity_label", data=graph_sentiment)
plt.title("Subjectivity vs Polarity")
plt.savefig('tweet_sentiment_plot.png')

# %%
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Create our contextual stop words
tfidf_stops = ["online", "class", "course", "learning", "learn","teach", "teaching", "distance", \
               "distancelearning", "education", "teacher", "student", "grade", "classes", "computer", "resource", \
               "onlineeducation", "onlinelearning", "school", "students", "class", "virtual", "eschool", "thing", \
               "virtuallearning", "educated", "educates", "teaches", "studies", "study", "semester", "elearning", \
               "teachers", "lecturer", "lecture", "amp", "academic", "admission", "academician", "account", "action",\
               "add", "app", "announcement", "application", "adult", "classroom", "system", "video", "essay", "training", \
               "homework","work","assignment", "paper", "get", "math", "project", "science", "physics", "lesson", "schools", \
               "courses", "assignments", "know", "instruction","email", "discussion","home", "college", "exam", "university", \
               "use", "fall", "term", "proposal", "one", "review", "proposal", "calculus", "search", "research", "algebra", \
               "internet", "remote", "remotelearning"]

# Initialize a Tf-idf Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words= tfidf_stops)

# Fit and transform the vectorizer
tfidf_matrix = vectorizer.fit_transform(tokens.str.join(" "))



# Create a new DataFrame called frequencies
frequencies = pd.DataFrame(tfidf_matrix.sum(axis=0).T,index=vectorizer.get_feature_names(),columns=['total frequency'])

# Sort the words by frequency
frequencies.sort_values(by='total frequency',ascending=False, inplace=True)

# Join the indexes
frequent_words = " ".join(frequencies.index)+" "

# Initialize the word cloud
wc = WordCloud(width = 1920, height = 1080, min_font_size = 10, max_words=2000, background_color ='white', stopwords= tfidf_stops)

# Generate the world clouds for each type of label
tweets_wc = wc.generate(frequent_words)

# Plot the world cloud                     
plt.figure(figsize=(16,9), facecolor = None) 
plt.imshow(tweets_wc, interpolation="bilinear") 
plt.axis("off") 
plt.title("Common words in the tweets")
plt.tight_layout(pad = 0) 
# plt.show()
plt.savefig("tweet_word_cloud_plot.png")


# %%
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

train_data, test_data, y_train, y_test = train_test_split(
    tokens, df["tweet_polarity_label"].values, test_size=0.33, random_state=1000
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)

X_train = tokenizer.texts_to_sequences(train_data)
X_test = tokenizer.texts_to_sequences(test_data)

y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values


# %%
from keras.preprocessing.sequence import pad_sequences

maxlen = max([len(x) for x in X_train])

X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

word_index = tokenizer.word_index

# %%
import gensim

embedding_dim = 50
model = gensim.models.Word2Vec(
    sentences=tokens, vector_size=embedding_dim, window=5, workers=6, min_count=1
)


# %%
filename = "tweet_embedding_word_vector.txt"
model.wv.save_word2vec_format(filename, binary=False)


# %%
import os
import numpy as np

embeddings_index = {}
f = open(os.path.join("", "tweet_embedding_word_vector.txt"), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()


# %%
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# %%
from keras.models import Sequential
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
# from keras.layers.wrappers import Bidirectional

model = Sequential()
model.add(
    Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=maxlen,
        trainable=False,
    )
)
model.add(LSTM(128, recurrent_dropout=0.5))
model.add(Dense(3, activation="softmax"))

model.summary()


# %%
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# %%
model_history = model.fit(
    X_train, y_train, epochs=50, validation_data=(X_test, y_test)
)


# %%

acc = model_history.history["accuracy"]
val_acc = model_history.history["val_accuracy"]
loss = model_history.history["loss"]
val_loss = model_history.history["val_loss"]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, "b", label="Training accuracy")
plt.plot(val_acc, "r", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy Value")
plt.title("Training and validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, "b", label="Training loss")
plt.plot(val_loss, "r", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Training and validation loss")
plt.legend()

plt.show()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "sans",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

plt.savefig('results.pgf')


# %%
from sklearn.metrics import confusion_matrix

y_pred_class = model.predict(X_test)
cf_matrix = confusion_matrix(
    y_test.argmax(axis=1), y_pred_class.argmax(axis=1).round(), labels=[0, 1, 2]
)
print(cf_matrix)


# %%
from sklearn.metrics import classification_report

y_pred_class = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred_class.argmax(axis=1).round()))


# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred_class.ravel())
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")


# %%
from sklearn import metrics

print(metrics.auc(fpr, tpr) * 100)


# %%
