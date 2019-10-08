#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You need to develop a classifier model for BIRD Audio Detection (BAD). This task is
# important for wildlife monitoring, especially with the increasing concerns on 
# climate changing and its effect on wildlife migration. The sound signal is decomposed
# into 256 Fourier components (features) to be able to detect the present of bird chirping.
#Less sound samples with bird chirping will indicate bird migration.

# This is a binary classification problem with two classes: BIRD chirp, non-BIRD noise 
# The data used in this assignment is derived from the BAD challenge. 


# In[110]:


import pandas as pd
# Importing sklearn machine learning models and packages
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
np.set_printoptions(threshold=np.inf)


# In[111]:


# For help with logistic regression implementation see below
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


# In[112]:


data = pd.read_csv('features.csv')

print ('There are', data.shape[0], 'rows/samples and', data.shape[1], 'columns/features in data set')


# In[ ]:





# In[113]:


# DO NOT RUN THIS CELL UNLESS DOWNSAMPLING IS REQUESTED

################################# Down sampling data dimension #########################
# Just take 10 evenly sampled features from 256 columns including the output 'class' column
idx = np.linspace(1, data.shape[1] - 1, 11).astype(int)

# Use the evenly spaced column indexes (idx) to select n evenly spaced columns from data matrix
myData = data [data.columns[idx]]

#################################################################
myData.head()


# In[114]:


# Split the dataset into training data and test data 
# 0.2 means 80%-20% split - test data 20% 

# 
train_set, test_set = train_test_split (myData, test_size =0.2, random_state = 42)

print ('After 80-20 split of the dataset, ')
print ('Training dataset has ', train_set.shape[0], 'samples or rows')
print ('Test dataset has ', test_set.shape[0], 'samples or rows')

print ('Number of columns or features in both sets is', train_set.shape[1])


# In[99]:


# Preparing training, testing data matrices and their labels.

# Encoding the categorical labels using integers, Bird (y=1) versus non-bird (y=0)
# Training labels (output binary labels) - used for supervised learning/training
ytr = (train_set['class']=='BIRD').astype(int)

# Test labels (output binary labels)- will be used to evaluate model accuracy only
yts = (test_set['class']=='BIRD').astype(int)

# Training data matrix Xtr excluding the first two columns because they are just labels
xtr = train_set.iloc[:, 2:]

# Test data matrix Ytr excluding the first two columns because they are just labels
xts = test_set.iloc[:, 2:]


# In[115]:


# Training the model, C is inverse of lambda, the regularization term

clf = LogisticRegression(penalty='l2', C = 0.1).fit(xtr, ytr)

print('After training, intercept is, wo = ', clf.intercept_[0])


print ('After training, weights of n features are as follows.')

weightVec = clf.coef_[0] # Is a matrix inside a matrix, so pulling that out

for t, col_name in enumerate (list(xtr)):
    
    print('The weight of', col_name, 'is', weightVec[t])


# In[116]:


# Liklihood probability of test features, output of phi(z)
# For two possible outcomes Bird and No Bird, there will be two columns
# The first column represents the liklihood of being no bird and the second of being bird
yPredict_ts = clf.predict_proba(xts)

# you may want to print the original yPredict_ts to make sense of the output matrix

#print (yPredict_ts)

for p in range(len(yPredict_ts)) :
    
    yPred = np.round(yPredict_ts[p],2) # rounding up to two decimal places
    print ('The likelihood of test sample ',p,'being non-BIRD is',yPred[0],'and BIRD is',yPred[1] )


# In[117]:



# predicting actual label, 0 or 1 on test data
# Recall label 1 means BIRD and label 0 means non-BIRD

yhat_ts = clf.predict(xts)

for k in range (len(yts)):
    print ('Actual', yts.values[k], 'label was predicted as', yhat_ts [k], 'label')

   


# In[118]:




# You can use scikit learn package to calculate the accuracy by comparing predicted Yts and
# actual Yts
accuracy_score(yts, yhat_ts)

# Or you can hand calculate using your own formula
acc = np.sum(yts==yhat_ts)*100/len(yts)

print ('The accuracy in classifying between bird chirp and non-bird sound is:', acc,'%')


# In[90]:


xfea = 'F178'
yfea = 'F203'

dataSt = train_set[[xfea,yfea]].values

# Create a mesh to plot in
x_min, x_max = dataSt[:,0].min() , dataSt[:,0].max() 
y_min, y_max = dataSt[:,1].min() , dataSt[:,1].max() 


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))

clf_DB = LogisticRegression().fit(dataSt, ytr)


# In[91]:



print ('Weights are:', clf_DB.coef_)
print ('Intercept is:', clf_DB.intercept_)
print ('The decision boundary equation:')

print ('y = ', clf_DB.intercept_[0], clf_DB.coef_[0][0], 'XF178 +', clf_DB.coef_[0][1], 'XF203' )

bird_class = np.where(ytr == 1)
nbird_class = np.where(ytr == 0)

Z = clf_DB.predict(np.c_[xx.ravel(), yy.ravel()])
#
# Put the result into a color plot
Z = Z.reshape(xx.shape)



axis_font = {'fontname':'Arial', 'size':'20'}
fig_01 = plt.figure(figsize=(10,10)) 

#ax = 
plt.contourf(xx, yy, Z, cmap=plt.cm.gray, alpha=0.2)

#ax =plt.imshow(Zx,cmap ='gray')

plt.scatter(dataSt [bird_class,0], dataSt[bird_class,1], s=60, edgecolors='black',
              facecolors='red', linewidths=1, label='Bird')
plt.scatter(dataSt[nbird_class,0], dataSt[nbird_class,1], s=60,edgecolors='black',
               facecolors='', marker ='s', linewidths=1, label='Not Bird')

# Plot also the training points
#plt.scatter(dataSt[:, 0], dataSt[:, 1], c='green')
plt.xlabel(xfea,**axis_font)
plt.ylabel(yfea, **axis_font)
#plt.xlim(40, 75)
plt.ylim(yy.min(), yy.max())

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


# Draw Legend
leg = plt.legend(loc='upper right', ncol=1, mode="", shadow=False, fontsize = 20, fancybox=True)


# In[ ]:




