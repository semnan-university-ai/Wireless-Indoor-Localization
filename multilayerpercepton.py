#!/usr/bin/env python
# coding: utf-8

# ### Multi layer percepton

# In[1]:


import pandas as pd
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


# In[2]:


dataset = pd.read_csv('wifi_localization.csv', sep='\t' , header=0)
dataset


# In[3]:


#dataset.head()
#dataset.shape
#dataset.info()
#print(dataset['t'])
#y=(dataset['t'])
#x=dataset[['a','b','c','d','e','f','g','h','i','j','k','l']]
#print(x)
#print(y)
#dataset.hist(bins=50, figsize=(20,15))
#p.show()
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=40)
#train_set.shape
#test_set.shape
#train_set.head
train_set_att = train_set.drop(['lable'], axis=1)
train_set_t = train_set['lable']
test_set_att = test_set.drop(['lable'], axis=1)
test_set_t = test_set['lable']


print("x_train: ",train_set_att.shape)
print("x_test: ",test_set_att.shape)
print("y_train: ",train_set_t.shape)
print("y_test: ",test_set_t.shape)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(train_set_att, train_set_t.values.ravel())
y_pred = mlp.predict(test_set_att)
print("Accuracy:",metrics.accuracy_score(test_set_t, y_pred))

