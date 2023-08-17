#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[11]:


np.random.seed(500)


# In[12]:


Corpus = pd.read_csv(r"C:\Users\yeten\Downloads\flavors.csv",encoding='latin-1')


# In[14]:


Corpus.columns


# In[17]:


Corpus['celery'].dropna(inplace=True)
Corpus['celery'] = [entry.lower() if isinstance(entry, str) else np.nan for entry in Corpus['celery']]
Corpus['celery'] = [word_tokenize(entry) if isinstance(entry, str) else [] for entry in Corpus['celery']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['celery']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus.loc[index,'celery_final'] = str(Final_words)


# In[18]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['celery_final'],Corpus['vegetable'],test_size=0.3)


# In[19]:


Encoder=LabelEncoder()
Train_Y=Encoder.fit_transform(Train_Y)
Test_Y=Encoder.fit_transform(Test_Y)


# In[22]:


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['celery_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[23]:


print(Tfidf_vect.vocabulary_)


# In[24]:


print(Train_X_Tfidf)


# In[25]:


Naive=naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
prediction_NB=Naive.predict(Test_X_Tfidf)
print("naive bayes accuracy score->",accuracy_score(prediction_NB,Test_Y)*100)


# In[26]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[ ]:





# In[ ]:




