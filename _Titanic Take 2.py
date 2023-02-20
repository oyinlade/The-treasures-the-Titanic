#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')
train.head()


# In[2]:


train.dtypes


# In[3]:


train.describe()


# In[4]:


train.info()


# In[5]:


import matplotlib.pyplot as pt
sex_pivot = train.pivot_table(index = 'Sex', values = 'Survived')
sex_pivot.plot.bar()
pt.show()


# Females survived more than the males (wayyyy more). Is there a science behind this?

# In[8]:


import matplotlib.pyplot as pt
sex_pivot = train.pivot_table(index = 'Embarked', values = 'Survived')
sex_pivot.plot.bar()
pt.show()


# C - Cherboug, Q - Queenstown, S - Southampton
# 
# Question - Why did people who embarked in Cherboug survive more than others?

# In[33]:


train['Age'].describe()


# In[ ]:





# In[34]:


survived =train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
survived['Age'].plot.hist(alpha = 0.5, color ='red', bins = 50)
died['Age'].plot.hist(alpha = 0.5, color ='blue', bins = 50)
pt.legend(['Survived', 'Died'])
pt.show()


# The lower the age, the greater the survival. What's the science behind this?

# In[ ]:


survived =train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
survived['Age'].plot.hist(alpha = 0.5, color ='red', bins = 50)
died['Age'].plot.hist(alpha = 0.5, color ='blue', bins = 50)
pt.legend(['Survived', 'Died'])
pt.show()


# In[35]:


def process_age(df,cut_points,label_names):
   df["Age"] = df["Age"].fillna(-0.5)
   df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
   return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()
pt.show()


# In[36]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


lr = LogisticRegression()


# In[39]:


columns = ['Pclass_2', 'Pclass_3', 'Sex_male']
lr.fit(train[columns], train['Survived'])


# Why use Pclass 2 and 3? Is it because of their values?

# In[40]:


from sklearn.linear_model import LogisticRegression

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
      'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])


# In[41]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)


# In[42]:


holdout = test  

from sklearn.model_selection import train_test_split, cross_val_score

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)


# In[43]:


lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)


# In[44]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)


# In[45]:


from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
print(accuracy)


# In[46]:


cross_val_score(estimator, X, y, cv=None)


# In[47]:


from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)


# In[48]:


lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])


# In[49]:


holdout_ids = holdout["PassengerId"]
sublesson_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
sublesson = pd.DataFrame(sublesson_df)


# In[50]:


sublesson.to_csv("sublesson.csv",index=False)


# In[ ]:




