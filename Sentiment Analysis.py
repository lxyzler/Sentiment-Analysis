#%%
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.corpus import stopwords
from subprocess import check_output
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download("stopwords")
data = pd.read_csv('/home/joker/Desktop/5002/Q5/train.csv')
test = pd.read_csv('/home/joker/Desktop/5002/Q5/test1.csv')
data['SentimentText'] = data['SentimentText'].str.replace("[^a-zA-Z#]", " ")    
data['SentimentText'] = data['SentimentText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))  

traindata = []
stopwords_set = set(stopwords.words("english"))
for index, row in data.iterrows():
    words_filtered = [e.lower() for e in row.SentimentText.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    traindata.append(words_without_stopwords)

s = []
y=[]
for i in range(len(data['Sentiment'])):
    y.append(data['Sentiment'].iloc[i])
    if data['Sentiment'].iloc[i]==0:
        s.append('Negative')
    else:
        s.append('Positive')

sentences = []
for i in range(len(traindata)):
    sentences.append(TaggedDocument(traindata[i],s[i]))
sentences


import random
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7) #instantiate a Doc2Vec model 
model.build_vocab(sentences)  #build a vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  #train model
model.most_similar('good')
model[traindata[0][0]]
model.infer_vector(traindata[0])
train = []
for i in range(len(traindata)):
    train.append(model.infer_vector(traindata[i]))


testdata = []
stopwords_set = set(stopwords.words("english"))

for index, row in test.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    testdata.append(words_without_stopwords)
test = []
for i in range(len(testdata)):
    test.append(model.infer_vector(testdata[i]))

for i in range(len(s)):
    if s[i]== 'Positive':
        s[i] = 1
    else: 
        s[i] = 0
from sklearn.ensemble import RandomForestClassifier
param_test1= {'n_estimators':range(10,71,100)}
from sklearn.grid_search import GridSearchCV
gsearch1= GridSearchCV(estimator = RandomForestClassifier(max_depth=6,random_state=50, min_samples_split=2, min_samples_leaf=1),
                       param_grid =param_test1,cv=5)
gsearch1.fit(train,y)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)
clf = RandomForestClassifier(n_estimators=100, max_depth=6,random_state=50, min_samples_split=2, min_samples_leaf=1)
clf.fit(train, y)

output=clf.predict(test)

x = pd.read_csv('/home/joker/Desktop/5002/Q5/test1.csv')
out=pd.DataFrame(output)
re=pd.concat([x,out],axis=1)
re.columns=['Contents','Result']
re.to_csv('​ Q5_output.csv',index=True,index_label='ID')

