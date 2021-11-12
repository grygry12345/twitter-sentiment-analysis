# Twitter Sentiment Analysis for Distance Learning

Coronavirus has been affected all over the world. Also, education methods have changed since this disease occurred. Social media are the essential tools that show users' reactions, and Twitter is one of the most popular social media as text data. Therefore tweets are valuable data for use in data analysis. In addition, it could be used in AI by supervised learning method with embedded word vector.

- [Project Presentation](https://1drv.ms/b/s!AjL8ixsEfu21gf0TKg171H0DqkuGHg?e=5Xl0mP 'Presentation')

- [Project Report](https://1drv.ms/b/s!AjL8ixsEfu21gfgDd5Ho4TQDxkKpJQ?e=Cx5Aju 'Report')

## Feature Selection

38 columns 19 are removed 19 selected

```csv
,id,
conversation_id,
created_at,
date,
tweet,
hashtags,
cashtags,
user_id,
user_id_str,
username,
name,
urls,
photos,
video,
retweet,
nlikes,
nreplies,
nretweets,
reply_to
```

## Results

Tweets are mostly positive or neutral due to commercials in tweets. Therefore negative tweets are training models that could not be as accurate as neutral or positive tweets.

`y` seeperated three different outputs `0` negative `1` neutural `2` positive.

AUC = 97.79

| |precision | recall | f1-score | support |
|---------|---------|----------|---------|---------|
 0 | 0.78 | 0.69 | 0.74 | 2455 |
 1 | 0.88 | 0.94 | 0.91 | 8575 |
 2 | 0.95 | 0.92 | 0.93 | 12162 |
   |  |  |  |  |
  accuracy | | | 0.90 | 23192 |
  macro avg | | | 0.90 | 23192 |
  weighted avg | | | 0.90 | 23192 |
