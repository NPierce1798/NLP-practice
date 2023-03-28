#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from tqdm.notebook import tqdm



plt.style.use('ggplot')


#setting df, only using the first 500 rows, we don't need all of them for this
df = pd.read_csv('Reviews.csv')
df = df.head(500)


#doing some EDA

df['Score'].value_counts()

#There are mostly 5 star reviews
'''5: 339
   4: 70
   3: 37
   2: 18
   1: 36'''


#basic NLKT

example = df['Text'][50]
print(example)


print('//')
#creating a tokenized version of the string
tokens = nltk.word_tokenize(example)
print(tokens[:10])


print('//')
#showing the part of speech for the tokenized string
print(nltk.pos_tag(tokens))

print('//')
#we can tag these set to a variable

tagged = nltk.pos_tag(tokens)
print(tagged[:10])


print('//')
#we can chunk these together on their tags
chunked = nltk.chunk.ne_chunk(tagged)
chunked.pprint()



#now using VADER for sentiment analysis


#first instantiate
sia = SentimentIntensityAnalyzer()

#polarity score returns the sentiment of a string
print(sia.polarity_scores('I am so happy'))

#we can get the sentiment of our example string

print(sia.polarity_scores(example))


#now we want to run the polarity score on the whole df for overall sentiment
results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    id = row['Id']

    results[id] = sia.polarity_scores(text)
    

#now we want to turn our results dictionary into a dataframe

vaders = pd.DataFrame(results).T    # .T orients it correctly

vaders = vaders.reset_index()#.rename(columns={'Index':'Id'})    #resetting the index, setting it to Id

print(vaders)
#now merge it to our original df

vaders = vaders.merge(df, left_index=True, right_index=True, how='left')

#we can print the df and the vaders to check our merge
print(df.info())
print(vaders.info())


#to help visualize the correlation between the score and rating, we can plot them

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound score by review stars')
plt.show()  #to show in vs code, add #%% at the top of the code


#we can compare how sentiment varies by rating:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('positive')
axs[1].set_title('neutral')
axs[2].set_title('negative')
plt.show()