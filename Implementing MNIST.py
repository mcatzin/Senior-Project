#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


mnist = fetch_mldata(dataname = "MNIST original")


# In[6]:


x = mnist['data']
y = mnist['target']


# In[7]:


print(x.dtype, y.dtype)
print(x.shape, y.shape)


# In[14]:


def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) //n_cols
    fig = plt.figure(figsize = (8, 8))
    
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i +1)
        plt.axis("off")
        plt.imshow(images[i], cmap=plt.cm.gray)
        sp.set_title(labels[i])
    plt.show()
    
#lets plot random 20 images
p = np.random.permutation(len(x))
p = p[:20]
plot_images(x[p].reshape(-1, 28, 28),y[p])


# In[15]:


y = y.astype("int32")
x = x / 255.0


# In[17]:


x.min(), x.max()


# In[19]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y)
#train 70% and test 30%


# In[20]:


train_x.shape, test_x.shape


# In[21]:


#train model
from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB()
cls.fit(train_x, train_y)


# In[22]:


#Evaluate model

cls.score(test_x, test_y)


# In[23]:


from sklearn.metrics import classification_report
predictions = cls.predict(test_x)
print(classification_report(test_y, predictions))


# In[24]:


p = np.random.permutation(len(test_x))
p = p[:20]
plot_images(test_x[p].reshape(-1,28, 28), predictions[p])


# In[25]:


p = np.random.permutation(len(test_x))
p = p[:20]
plot_images(test_x[p].reshape(-1,28, 28), predictions[p])


# In[ ]:




