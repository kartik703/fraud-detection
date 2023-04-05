#!/usr/bin/env python
# coding: utf-8

# In[252]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[253]:


df=pd.read_csv("loan.csv")


# In[254]:


df.shape


# In[255]:


df.head()


# In[256]:


df.tail()


# In[257]:


a=df.columns
for i in a:
  print(i)


# In[258]:


df=df.iloc[:,:53]


# **Data** **Preprocessing**

# In[259]:


null_per=(df.isnull().sum()/len(df))*100


# In[ ]:





# In[260]:


null_per


# In[261]:


filter=null_per[null_per>60]


# In[262]:


filter


# In[263]:


df = df.drop(filter.index, axis=1)


# In[264]:


df.isnull().sum()


# In[265]:


df


# In[266]:


df["home_ownership"].unique()


# In[267]:


df1=df.replace([' 36 months',' 60 months'],[30,60])
df1=df1.replace(['A','B','C','D','E','F','G'],[1,2,3,4,5,6,7])
df1=df1.replace(['Not Verified','Verified','Source Verified'],[0,1,2])
df1=df1.replace(['10+ years','< 1 year','1 year','3 years','8 years','9 years','4 years','5 years','6 years','2 years','7 years'],[10,0,1,3,8,9,4,5,6,2,7])                
df1=pd.get_dummies(df1,columns=['home_ownership'])               



# In[268]:


import re
def remove_per(text):
  re_pun=re.sub("[^0-9]","",str(text))
  if re_pun=="":
    re_pun=0
  return re_pun
df1["int_rate"] = df1["int_rate"].apply(remove_per)
df1["revol_util"] = df1["revol_util"].apply(remove_per)


# In[269]:


df1[df1["revol_util"]==""]


# In[270]:



df1


# In[270]:





# In[271]:


df2=df1.drop(["id","member_id","sub_grade","emp_title","pymnt_plan","url","desc","purpose","title","zip_code","addr_state","application_type","initial_list_status"],axis=1)


# In[272]:


df2


# In[273]:


import datetime


df2['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
df2['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y')
df2['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%y')
df2['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%y')


# In[274]:



df2['earliest_cr_line_year']=df2['earliest_cr_line'].dt.year
df2['issue_d_month']=df2['issue_d'].dt.month
df2['last_pymnt_d_month']=df2['last_pymnt_d'].dt.month
df2['last_credit_pull_d_month']=df2['last_credit_pull_d'].dt.month


# In[275]:


df2['issue_d'].iloc[:50]


# In[276]:


df2.iloc[:,40:]


# In[277]:


df2["loan_status"].value_counts()


# In[278]:



df2=df2.replace(['Charged Off','Fully Paid','Current'],[0,1,1])


# In[315]:


df2.iloc[:,15:]


# In[280]:


df2["loan_status"].value_counts()


# In[281]:


df2.isnull().sum()


# In[282]:


df2[df2["last_pymnt_d_month"].isnull()]["last_pymnt_d_month"]

df2['last_pymnt_d_month']=df2['last_pymnt_d_month'].fillna(0)
df2['last_credit_pull_d_month']=df2['last_credit_pull_d_month'].fillna(0)
df2['emp_length']=df2['emp_length'].fillna(0)


# In[ ]:





# In[283]:


df2.isnull().sum()


# In[284]:


df2=df2.drop(['earliest_cr_line','last_pymnt_d','last_credit_pull_d','issue_d','collections_12_mths_ex_med','policy_code'], 1)


# In[292]:


df2.sample(10)


# In[303]:



df2["loan_status"].value_counts().plot.pie(explode=[0.05,0],autopct="%2.1f%%")


# In[307]:



fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(data=df2, x="emp_length",ax=axes[0])
sns.histplot(data=df2, x="grade",ax=axes[1])


# In[324]:


plt.figure(figsize=(4, 4))
sns.distplot(df2["loan_amnt"])


# In[298]:


plt.figure(figsize=(15, 12))
sns.heatmap(df2.corr())


# In[294]:


df2.corr()


# **Splitting test and train data**

# In[285]:


x = df2.drop('loan_status', 1)
y = df2["loan_status"]


# In[218]:



x.info()


# In[219]:


y


# **Scaling using standard scaler**

# In[220]:


from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
x_scaled


# In[221]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y, random_state=2)


# **Logistic Regression**

# In[222]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


# In[223]:


y_logistic = lr_model.predict(X_test)

y_logistic


# In[227]:


from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(y_test, y_logistic)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(y_test, y_logistic))


# In[228]:


y_logistic1 = lr_model.predict(X_train)

y_logistic1


# In[229]:


from sklearn.metrics import accuracy_score
lr_accuracy =accuracy_score(y_test,y_logistic)
lr_accuracy


# In[230]:


accuracy_score(y_train,y_logistic1)


# **DecisionTree Classifier**

# In[235]:


from sklearn.tree import DecisionTreeClassifier
d_model = DecisionTreeClassifier().fit(X_train, y_train)
d_predictions = d_model.predict(X_test)
d_predictions


# In[237]:


d_predictions1 = d_model.predict(X_train)
d_predictions1


# In[238]:


from sklearn.metrics import accuracy_score
lr_accuracy =accuracy_score(y_test,d_predictions)
lr_accuracy


# In[239]:


from sklearn.metrics import accuracy_score
lr_accuracy =accuracy_score(y_train,d_predictions1)
lr_accuracy


# In[241]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, d_predictions)
matrix


# **Random Forest Classifier**

# In[242]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 40)
classifier.fit(X_train, y_train)


# In[244]:


r_predictions=classifier.predict(X_test)
r_predictions


# In[246]:


accuracy_score(y_test,r_predictions)


# In[245]:


r_predictions1=classifier.predict(X_train)
r_predictions


# In[247]:


accuracy_score(y_train,r_predictions1)


# In[248]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, r_predictions)
matrix


# **Saving Model using pickle**

# In[250]:


import pickle
pickle.dump(lr_model, open('loan.pkl', 'wb'))
pickled_model = pickle.load(open('loan.pkl', 'rb'))
a=pickled_model.predict(X_test)


# In[251]:


accuracy_score(y_test,a)


# In this project I could identify that almost all algorithm that I had used are giving good prediction. Here I am choosing Logistic Algorithm for saving my model as this gives better prediction than others.

# In[ ]:




