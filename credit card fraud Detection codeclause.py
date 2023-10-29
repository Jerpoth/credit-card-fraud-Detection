#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Importing the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[9]:


# Loading the Datasets

credit_card=pd.read_csv('D:\\DBMS\\notes\\ML\\creditcard.csv')


# In[10]:


# First 5 rows

credit_card.head()


# In[11]:


credit_card.tail()


# In[12]:


# Data set information

credit_card.info()


# In[13]:


# checking the missing Value in the column

credit_card.isnull().sum()


# In[14]:


# Distribution of legit transaction & fraudulent transaction

credit_card['Class'].value_counts()


# The majority of transactions in the dataset are classified as non-fraudulent, accounting for approximately 99.83% of the total transactions.
# The significant imbalance in class distribution highlights the challenge of detecting fraudulent transactions, as they are relatively rare compared to non-fraudulent transactions.
# 

# In[30]:


# Plot histograms of each parameter 
credit_card.hist(figsize = (20, 20))
plt.show()


# In[15]:


# Separating the Data for Analysis

legit = credit_card[credit_card.Class == 0]
fraud = credit_card[credit_card.Class == 1]


# In[16]:


print(legit.shape)
print(fraud.shape)


# In[17]:


# Statistical measure of the data

legit.Amount.describe()


# In[18]:


fraud.Amount.describe()


# In[19]:


# Building a sample dataset containing Similar Distribution

legit_sample = legit.sample(n=492)


# In[20]:


# concating two Dataframe

new_dataset = pd.concat([legit_sample,fraud], axis=0)


# In[21]:


new_dataset.head()


# In[22]:


new_dataset.tail()


# In[23]:


new_dataset['Class'].value_counts()


# In[24]:


new_dataset.groupby('Class').mean()


# In[29]:


# Plot histograms of each parameter 
new_dataset.hist(figsize = (20, 20))
plt.show()


# In[34]:


# correlation matrix

corrmat = new_dataset.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)


# In[36]:


# splitting the data into feature and traget

x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']


# In[37]:


print(x)


# In[38]:


print(y)


# In[40]:


# spliting the dataset into trainging and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)


# In[46]:


print(x.shape,x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[43]:


#standardizing the dataset

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# Logestic Regression

# In[64]:


# Model Training

from sklearn.metrics import mean_squared_error,mean_absolute_error,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import math
logReg = LogisticRegression()
logReg.fit(x_train,y_train)
logReg_pred = logReg.predict(x_test)
logReg_acc = accuracy_score(y_test,logReg_pred)
logReg_mae = mean_absolute_error(y_test,logReg_pred)
logReg_mse = mean_squared_error(y_test,logReg_pred)
logReg_rmse= np.sqrt(logReg_mse)
logReg_prec= precision_score(y_test,logReg_pred)
logReg_recall= recall_score(y_test,logReg_pred)
logReg_f1 = f1_score(y_test,logReg_pred)


# In[65]:


print("Accuracy score for Logestic Regression is:", logReg_acc)
print("The Classification_report using Logestic Regression is :")
print(classification_report(y_test,logReg_pred))


# In[71]:


pip install xgboost


# XGBoost Classifier

# In[73]:


from xgboost import XGBClassifier
XGB=XGBClassifier()
XGB.fit(x_train,y_train)
XGB_pred=XGB.predict(x_test)
XGB_acc=accuracy_score(y_test,XGB_pred)
XGB_prec=precision_score(y_test,XGB_pred)
XGB_recall=recall_score(y_test,XGB_pred)
XGB_f1=f1_score(y_test,XGB_pred)


# In[75]:


print("The Accuracy of XGBoost is:",XGB_acc)
print("The Classifiation report of XGboost is:")
print(classification_report(y_test,XGB_pred))


# In[84]:


# Algorithm Compariion

models= pd.DataFrame({
    'Model': ['Logestic Regression','XGBoost'],
    'Accuracy': [logReg_acc,XGB_acc],
    'Precision': [logReg_prec,XGB_prec],
    'Recall':[logReg_recall,XGB_recall],
    'f1_score': [logReg_f1,XGB_f1]
})

models=models.sort_values(by='Accuracy',ascending=False)


# In[85]:


models

# According to the above table it is visible that Xgboost has performed a minute better than logistic regression in credit card fraud detection problem .
Hence for the precision we can go for XgBoost for its 95.65% and accuracy 93.34%
# In[ ]:




