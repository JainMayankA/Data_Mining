#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_json('../../data/train/trainText-75-25-bow.json', lines=True)


# In[2]:


df.head()


# In[3]:


recipes = list(df['recipe'])
labels = list(df['cuisine'])


# In[9]:


units = ['cup','cups','lb','oz','tablespoon','tablespoons', 'teaspoon', 'teaspoons', 'clove', 'cloves', 'small', 'large']
adjs = ['range', 'extra', 'corned', 'cooked', 'steamed', 'toasted', 'unseasoned','waxy','smoked','skim', 'shredded','seasoned', 'processed', 'peeled', 'organic', 'minced', 'chopped', 'peeled', 'drained', 'cut', 'ground', 'light', 'medium', 'melted', 'firm', 'neutral','lean', 'skinless', 'sliced', 'free', 'fine', 'granulated', 'packed', 'firmly', 'fresh', 'freshly']
stopwords = units + adjs + ['style', 'and', 'such', 'as', 'or', 'not', 'into', 'other', 'in', 'to']


# In[10]:


vectorizer = CountVectorizer(min_df=0.0001, stop_words = stopwords)
vectorizer.fit(recipes)
print('Number of Features: %d'%len(vectorizer.get_feature_names()))


# In[11]:


print(vectorizer.get_feature_names())


# In[12]:


X_train = vectorizer.transform(recipes).toarray()
Y_train = labels


# In[13]:


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, Y_train)


# In[22]:


input_recipe = input("What is your recipe? \n> ")
while input_recipe != 'quit':
    input_recipe = input_recipe.lower()
    X_input = vectorizer.transform([input_recipe]).toarray()
    prediction = clf.predict(X_input)[0]
    
    print('\nIdentified Ingredients: \n> %s'%vectorizer.inverse_transform(X_input))
    
    print('\nPredicted Cuisine Type: \n> %s'%prediction)
    print('\n====================================')
    
    input_recipe = input("What is your recipe? \n> ")


# In[15]:


from sklearn.pipeline import Pipeline
NB_model = Pipeline([('vectorizer', vectorizer),('NB', clf)])


# In[16]:


import pickle
filename = 'NB_model_bow_v1.sav'
pickle.dump(NB_model, open(filename,'wb'))


# In[21]:


NB_model.predict(list(['salmon rice', 'cheese pepperoni mushrooms']))


# In[ ]:




