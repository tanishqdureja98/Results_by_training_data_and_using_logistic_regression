#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv(r'C:\Users\hp\Downloads\Project_1.csv')
df


# In[4]:


df.dtypes


# In[5]:


##droping loan_id column
df.drop(['Loan_ID'],axis=1,inplace=True)
df


# In[6]:


##encoding columns
dummies=pd.get_dummies(df,columns=['Gender','Education','Property_Area'],prefix=['gender','education','property_area'])
dummies


# In[7]:


columns=['Gender','Education','Property_Area']
df=pd.concat([df,dummies],axis=1)
df.drop(columns,axis=1,inplace=True)


# In[8]:


df


# In[9]:


df = df.loc[:,~df.columns.duplicated()]
df


# In[10]:


df['Married']=df['Married'].replace({'Yes':1,'No':0})
df


# In[11]:


df['Self_Employed']=df['Self_Employed'].replace({'Yes':1,'No':0})
df


# In[12]:


df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})
df


# In[13]:


df['Dependents'] = df['Dependents'].replace({'3+':'3'})
df


# # filling missing values

# In[14]:


df.isnull().sum()


# In[15]:


median = df['Married'].median()
median


# In[16]:


df['Married'].fillna(median, inplace=True)
df


# In[17]:


df.isnull().sum()


# In[18]:


median1 = df['Dependents'].median()
median2 = df['Self_Employed'].median()
median3 = df['LoanAmount'].median()
median4 = df['Loan_Amount_Term'].median()
median5 = df['Credit_History'].median()


# In[19]:


df['Dependents'].fillna(median1,inplace=True)
df['Self_Employed'].fillna(median2,inplace=True)
df['LoanAmount'].fillna(median3,inplace=True)
df['Loan_Amount_Term'].fillna(median4,inplace=True)
df['Credit_History'].fillna(median5,inplace=True)
df


# In[20]:


df.isnull().sum()


# In[21]:


df.info()


# In[22]:


df['Dependents'].values


# In[23]:


df['Dependents']=df['Dependents'].astype(int)
df


# In[24]:


df.info()


# # training the data

# # logistic regression

# In[25]:


X=df.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]]
X


# In[26]:


y=df.iloc[:,[8]]
y


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)


# In[29]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[30]:


model.fit(X_train,y_train)


# In[31]:


y_predicted=model.predict(X_test)


# In[32]:


X_test


# In[33]:


y_predicted


# In[34]:


model.predict_proba(X_test)


# In[35]:


model.score(X_test,y_test)


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predicted)


# # random forest

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
regressor=RandomForestClassifier()
regressor.fit(X_train,y_train)


# In[39]:


y_pred=regressor.predict(X_test)
y_pred


# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[41]:


regressor.score(X_test,y_test)


# # decision tree

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)


# In[43]:


from sklearn import tree
model=tree.DecisionTreeClassifier()


# In[44]:


model.fit(X_train,y_train)


# In[45]:


y_pred=model.predict(X_test)
y_pred


# In[46]:


y_test


# In[47]:


model.score(X_test,y_test)


# # SVM

# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)


# In[49]:


from sklearn.svm import SVC
model=SVC()


# In[50]:


model.fit(X_train,y_train)


# In[51]:


model.predict(X_test)


# In[52]:


model.score(X_test,y_test)

