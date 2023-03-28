#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#here we can use a model already pretrained on twitter comments
model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)

df = pd.read_csv('Reviews.csv')
df = df.head(500)

example = df['Text'][50]

print(example)

#to run with the roberta model, we first want to tokenize

encoded_text = tokenizer(example, return_tensors='pt')
print(encoded_text)
#now we can run our model on it

output = model(**encoded_text)

#now we bring it to numpy:

scores = output[0][0].detach().numpy()

#and we can apply the softmax to those scores

scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}

print(scores_dict)


#now lets make a function to run the polarity score on the whole dataframe

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


roberta_results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        id = row['Id']

        roberta_results[id] = polarity_scores_roberta(text)
    except RuntimeError:
        print(f'Broke from id {id}')    #running the full size in this example will break the code 
             #because it is too large for the roberta model