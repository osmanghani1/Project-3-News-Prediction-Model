#!/usr/bin/env python
# coding: utf-8

# # About the Dataset
# ### 1. id:unique id for a news article
# ### 2. title:title for a news article
# ### 3. author:author of the news article
# ### 4. text:the text of the article; could be incomplete
# ### 5. label:a label that marks weather the news article is real or fake

# In[1]:


# 1: Fake News
# 0: real News


#  # Importing the Dependencies

# In[78]:


import numpy as np
import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[79]:


import nltk
nltk.download('stopwords')


# In[80]:


#printing the stopwords lin English
print(stopwords.words('english'))


# # Data Preprocessing

# In[81]:


#loading the Dataset to Pandas DataFrame
news_dataset=pd.read_csv('train.csv')
news_dataset.head()


# In[82]:


news_dataset.shape


# In[83]:


# counting the number of missing values in the Dataset
news_dataset.isnull().sum()


# In[84]:


# replacing th null values with empty string
news_dataset=news_dataset.fillna('')


# In[85]:


# merging the author name and news title
news_dataset['content']=news_dataset['author']+' '+news_dataset['title']


# In[86]:


print(news_dataset['content'])


# In[87]:


# seprating the data and the label
X= news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']


# In[88]:


print(X)
print(Y)


# # Stemming :
# #### Steming is the process of reducing a word to its Root word
# #### actor, actress, acting --> act

# In[40]:


port_stem = PorterStemmer()


# In[92]:


def steamming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[93]:


news_dataset['content']= news_dataset['content'].apply(steamming)


# In[94]:


print(news_dataset['content'])


# In[95]:


#seprating the data and label
X= news_dataset['content'].values
Y= news_dataset['label'].values


# In[96]:


print(X)


# In[97]:


print(Y)


# In[98]:


Y.shape


# In[99]:


#converting the textual data to numerical data
vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X) 


# In[100]:


print(X)


# # Spliting the dataset to training and test Data

# In[106]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# # Training the Model:Logistic Regression 

# In[107]:


model= LogisticRegression()


# In[108]:


model.fit(X_train, Y_train)


# # Evaluation
# ### Accuracy Score

# In[109]:


#accuracy score on the trainig data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[110]:


print('accuracy score of the trainig data: ',training_data_accuracy)


# In[111]:


#accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[112]:


print('accuracy score of the test data: ',test_data_accuracy)


# # Making a predictive system 

# In[116]:


X_new=X_test[1]
prediction=model.predict(X_new)
print(prediction)

if(prediction[0]==0):
    print('The news is Real')
else:
        print('The news is Fake')


# In[117]:


print(Y_test[1])


# In[ ]:




