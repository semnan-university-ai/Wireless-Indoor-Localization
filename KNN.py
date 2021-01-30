#!/usr/bin/env python
# coding: utf-8

# ### KNN

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split


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
train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=40)


train_set_att = train_set.drop(['lable'], axis=1)
print(train_set_att)
train_set_t = train_set['lable']
test_set_att = test_set.drop(['lable'], axis=1)
test_set_t = test_set['lable']
from sklearn.neighbors import KNeighborsClassifier
K = 1
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(train_set_att, train_set_t)
print("When K = {} neighnors , KNN test accuracy: {}".format(K, knn.score(test_set_att,test_set_t)))
print("When K = {} neighnors , KNN train accuracy: {}".format(K, knn.score(train_set_att, train_set_t)))

ran = np.arange(1,30)
train_list = []
test_list = []
for i,each in enumerate(ran):
    knn = KNeighborsClassifier(n_neighbors=each)
    knn.fit(train_set_att, train_set_t)
    test_list.append(knn.score(test_set_att,test_set_t ))
    train_list.append(knn.score(train_set_att,train_set_t ))
    
p.figure(figsize=[15,10])
p.plot(ran,test_list,label='Test Score')
p.plot(ran,train_list,label = 'Train Score')
p.xlabel('Number of Neighbers')
p.ylabel('fav_number/retweet_count')
p.xticks(ran)
p.legend()
print("Best test score is {} , K = {}".format(np.max(test_list), test_list.index(np.max(test_list))+1))
print("Best train score is {} , K = {}".format(np.max(train_list), train_list.index(np.max(train_list))+1))

