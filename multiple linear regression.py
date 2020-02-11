#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 18, 20


# In[2]:


df = pd.read_csv('50_Startups.csv')
df


# In[3]:


def convert_to_int(word):
    word_dict = {'New York':0, 'California':1, 'Florida':2}
    return word_dict[word]


# In[5]:


df['State'] = df['State'].apply(lambda df : convert_to_int(df))


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


df['Profit']
sns.distplot(df['Profit'], color='r')


# In[10]:


x = df.drop('Profit', axis = 1)
x


# In[11]:


x.hist()


# In[12]:


y = df['Profit']
y


# In[13]:


corr = df.corr()
corr


# In[14]:


sns.heatmap(corr)


# In[15]:


df_numerical = df.select_dtypes(include = ['float64', 'int64'])
df_numerical


# In[16]:


for i in range(0, len(df_numerical.columns), 4):
    sns.pairplot(data=df_numerical,
                x_vars=df_numerical.columns[i:i+4],
                y_vars=['Profit'])


# In[17]:


s = {'R&D Spend': ['sum'], 'Administration': ['sum'], 'Marketing Spend': ['sum'], 'Profit':['sum']}
df1 = df.groupby('State').aggregate(s)
df1.columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
df1


# In[18]:


df1 = df1.reset_index()
df1


# In[19]:


sns.barplot(x=df1.State, y=df1.Profit, data=df1)


# In[20]:


sns.barplot(x=df1.State, y=df1['R&D Spend'], data=df1)


# In[21]:


sns.barplot(x=df1.State, y=df1['Marketing Spend'], data=df1)


# In[22]:


sns.barplot(x=df1.State, y=df1['Administration'], data=df1)


# In[24]:


sns.boxplot(x='State', y='Profit', data=df)


# In[25]:


sns.boxplot(x='State', y='R&D Spend', data=df)


# In[26]:


sns.boxplot(x='State', y='Marketing Spend', data=df)


# In[27]:


sns.boxplot(x='State', y='Administration', data=df)


# In[28]:


sns.regplot(x="Administration", y="Profit", data=df_numerical)


# In[29]:


sns.regplot(x="Marketing Spend", y="Profit", data=df_numerical)


# In[30]:


sns.regplot(x="R&D Spend", y="Profit", data=df_numerical)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()


# In[33]:


lr = model.fit(X_train, y_train)


# In[34]:


# Saving model to disk
import pickle
pickle.dump(lr, open('model.pkl','wb'))


# In[39]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[150906, 72456, 62321, 1]]))


# In[ ]:




