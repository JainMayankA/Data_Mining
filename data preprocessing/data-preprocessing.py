#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_json('datasets/recipe/train.json')


# In[2]:


df.head()


# In[3]:


def build_recipe(row):
    ingredients = row['ingredients']
    if len(ingredients) == 0:
        return ''
    else:
        recipe = ingredients[0]
        for i in range(1,len(ingredients)):
            recipe = recipe + ' ' + ingredients[i].strip().replace(' ','_').strip('_')
        return recipe


# In[4]:


label_mapping = {}
label = 0
for cuisine in df['cuisine'].unique():
    label_mapping[cuisine] = label
    label += 1
def label_cuisine(row):
    return label_mapping[row['cuisine']]


# In[5]:


df['recipe'] = df.apply(build_recipe, axis=1)
df['label'] = df.apply(label_cuisine, axis=1)


# In[6]:


df.head()


# In[7]:


recipes = df['ingredients'].tolist()
recipes[:1]


# In[8]:


ingredients = {}
for lst in df['ingredients']:
    for i in lst:
        i = i.replace(' ','_').strip('_')
        if not i in ingredients:
            ingredients[i] = 1
        else:
            ingredients[i] += 1


# In[9]:


ingredients['black_olives']


# In[10]:


MIN_FREQUENCY = 10
filtered_ingredients = {k: v for k, v in ingredients.items() if v >= MIN_FREQUENCY}


# In[11]:


recipes = list(df['recipe'])
labels = list(df['cuisine'])


# In[12]:


vectorizer = CountVectorizer()
vectorizer.fit(filtered_ingredients.keys())
print('Number of Features: %d'%len(vectorizer.get_feature_names()))


# In[13]:


print(vectorizer.get_feature_names())


# In[14]:


X_train = vectorizer.transform(recipes).toarray()
Y_train = labels


# In[15]:


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, Y_train)


# In[17]:


input_recipe = input("What is your recipe? \n> ")
while input_recipe != 'quit':
    input_recipe = input_recipe.lower()
    X_input = vectorizer.transform([input_recipe]).toarray()
    prediction = clf.predict(X_input)[0]
    
    print('\nIdentified Ingredients: \n> %s'%vectorizer.inverse_transform(X_input))
    
    print('\nPredicted Cuisine Type: \n> %s'%prediction)
    print('\n====================================')
    
    input_recipe = input("What is your recipe? \n> ")


# In[ ]:


from sklearn.pipeline import Pipeline
NB_model = Pipeline([('vectorizer', vectorizer),('NB', clf)])


# In[ ]:




