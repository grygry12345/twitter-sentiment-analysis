# %%
import pandas as pd
import re

from torch.backends.mkldnn import flags

wordVector = pd.read_csv("./tweet_embedding_word_vector.txt", sep=" ", header=None)
unfoundTable = pd.read_csv("./tweet_unfoundWord.csv")

valueErrors = []
isAbbrevation = re.compile("[_]")

for index, row in wordVector.iterrows():
    # Chack if vector is not abbrevation or phrase
    if isAbbrevation.search(str(row[0])) == None:
        regex = "^" + str(row[0]) + "$"
        if unfoundTable["0"].str.contains(regex).any():
            valueErrors.append(row[0])
# %%

print(len())

# %%

