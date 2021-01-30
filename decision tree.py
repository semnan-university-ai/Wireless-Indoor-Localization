#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree

# In[1]:


import pandas as pd
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neural_network import MLPClassifier


# In[2]:


dataset = pd.read_csv('wifi_localization.csv', sep='\t' , header=0)
dataset


# In[5]:


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
train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=40)
#train_set.shape
#test_set.shape
#train_set.head
train_set_x = train_set.drop(['lable'], axis=1)
train_set_y = train_set['lable']
test_set_x = test_set.drop(['lable'], axis=1)
test_set_y = test_set['lable']


print("x_train: ",train_set_x.shape)
print("x_test: ",test_set_x.shape)
print("y_train: ",train_set_y.shape)
print("y_test: ",test_set_y.shape)


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(train_set_x, train_set_y.ravel())
print("accuracy: ", dtree.score(test_set_x, test_set_y))
p.figure(figsize=(35,35))
temp = tree.plot_tree(dtree, fontsize=12)
p.show()

