
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd
import sys
from sklearn.tree import DecisionTreeClassifier


# In[35]:


my_data = pd.read_csv("training_2.csv", delimiter=",")


# In[36]:


#my_data= my_data[0:1000]


# In[37]:



#sys.getsizeof(my_data)/(1024*1024*1024)
#np.unique(my_data["PROV_B_M_A_STATE_NAME"])


# In[38]:


X = my_data[['ENTITY_TYPE_CODE','PROVIDERNAME', 'PROV_B_M_A_STATE_NAME', 'PROV_B_M_A_POSTAL_CODE', 'SPECDESC','PROV_B_P_B_P_LOC_ADD_TEL_NUM','PROV_B_M_A_CITY_NAME','PRIMARY_TAXONOMY']].values
#X[0:3]


# In[39]:


from sklearn import preprocessing

le_PROVIDERNAME = preprocessing.LabelEncoder()
le_PROVIDERNAME.fit(np.unique(my_data["PROVIDERNAME"]))
X[:,1] = le_PROVIDERNAME.transform(X[:,1]) 

le_STATE_NAME = preprocessing.LabelEncoder()
le_STATE_NAME.fit(np.unique(my_data["PROV_B_M_A_STATE_NAME"]))
X[:,2] = le_STATE_NAME.transform(X[:,2]) 

le_SPECDESC = preprocessing.LabelEncoder()
le_SPECDESC.fit(np.unique(my_data["SPECDESC"]))
X[:,4] = le_SPECDESC.transform(X[:,4]) 

le_CITY_NAME = preprocessing.LabelEncoder()
le_CITY_NAME.fit(np.unique(my_data["PROV_B_M_A_CITY_NAME"]))
X[:,6] = le_CITY_NAME.transform(X[:,6]) 


le_PRIMARY_TAXONOMY = preprocessing.LabelEncoder()
le_PRIMARY_TAXONOMY.fit(np.unique(my_data["PRIMARY_TAXONOMY"]))
X[:,7] = le_PRIMARY_TAXONOMY.transform(X[:,7]) 

 





# In[40]:


#X[0:100]


# In[41]:


y = my_data["NPI"]
y[0:5]


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
X_trainset


# In[44]:


X_trainset.shape


# In[45]:


y_trainset.shape


# In[46]:


npiTree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
npiTree # it shows the default parameters


# In[47]:


npiTree.fit(X_trainset,y_trainset)


# In[48]:


predTree = npiTree.predict(X_testset)


# In[49]:


print (predTree [0:5])
print (y_testset [0:5])


# In[50]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[51]:


np.take(my_data,np.random.permutation(my_data.shape[0]),axis=0)


# In[84]:


input = pd.read_csv("test2.csv", delimiter=",")


# In[85]:


input


# In[88]:



X_1 = input[['ENTITY_TYPE_CODE','PROVIDERNAME', 'PROV_B_M_A_STATE_NAME', 'PROV_B_M_A_POSTAL_CODE', 'SPECDESC','PROV_B_P_B_P_LOC_ADD_TEL_NUM','PROV_B_M_A_CITY_NAME','PRIMARY_TAXONOMY']].values
X[0:3]


# In[89]:


from sklearn import preprocessing

le_PROVIDERNAME = preprocessing.LabelEncoder()
le_PROVIDERNAME.fit(np.unique(my_data["PROVIDERNAME"]))
X_1[:,1] = le_PROVIDERNAME.transform(X_1[:,1]) 

le_STATE_NAME = preprocessing.LabelEncoder()
le_STATE_NAME.fit(np.unique(my_data["PROV_B_M_A_STATE_NAME"]))
X_1[:,2] = le_STATE_NAME.transform(X_1[:,2]) 

le_SPECDESC = preprocessing.LabelEncoder()
le_SPECDESC.fit(np.unique(my_data["SPECDESC"]))
X_1[:,4] = le_SPECDESC.transform(X_1[:,4]) 

le_CITY_NAME = preprocessing.LabelEncoder()
le_CITY_NAME.fit(np.unique(my_data["PROV_B_M_A_CITY_NAME"]))
X_1[:,6] = le_CITY_NAME.transform(X_1[:,6]) 


le_PRIMARY_TAXONOMY = preprocessing.LabelEncoder()
le_PRIMARY_TAXONOMY.fit(np.unique(my_data["PRIMARY_TAXONOMY"]))
X_1[:,7] = le_PRIMARY_TAXONOMY.transform(X_1[:,7]) 

 


# In[90]:


predTree = npiTree.predict(X_1)


# In[94]:


predTree


# In[95]:



df = pd.DataFrame (predTree)

filepath = 'file.txt'
df.savetxt(filepath, index=False)


# In[96]:


np.savetxt(r'np1.txt', df.values, fmt='%d')

