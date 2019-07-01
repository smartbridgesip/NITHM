#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


os.chdir("C:/Users/MAHENDRA/Desktop/DATA SET1")
os.getcwd()


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


df=pd.read_csv('t3-fertilizer4.csv')


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.cov()


# In[8]:


df.columns


# In[9]:


df.dtypes


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.corr()


# In[13]:


x=df.iloc[:,0:8].values


# In[14]:


x


# In[15]:


y=df.iloc[:,8].values


# In[16]:


y


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[18]:


lb=LabelEncoder()
x[:,4]=lb.fit_transform(x[:,4])
x[:,3]=lb.fit_transform(x[:,3])


# In[19]:


x


# In[20]:


x.shape


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


lb2=LabelEncoder()
y=lb2.fit_transform(y)


# In[23]:


y


# In[24]:


from sklearn.preprocessing import OneHotEncoder


# In[25]:


onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]


# In[26]:


x


# In[27]:


x.shape


# In[28]:


y=y.reshape(-1,1)
y.ndim


# In[44]:


import statsmodels.api as sm
model1=sm.OLS(y,x).fit()
predictions=model1.predict()
model1.summary()


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


dt=DecisionTreeClassifier()


# In[33]:


dt.fit(x_train,y_train)


# In[34]:


y_pred=dt.predict(x_test)


# In[35]:


from sklearn.metrics import accuracy_score


# In[36]:


accuracy_score(y_test,y_pred)


# In[46]:





# In[47]:





# In[52]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




