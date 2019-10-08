#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


# 2D matri
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[4]:


def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) // n_cols
    fig = plt.figure(figsize = (8,8))
    
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i +1)
        plt.axis("off")
        plt.imshow(images[i], cmap=plt.cm.gray)
        sp.set_title(labels[i])
        
    plt.show()


# In[ ]:


p = np.random.permutation(len(x_train))
p = p[:20]
plot_images(x_train[p], y_train[p])


# In[9]:


#creating Fully connected neural by converting 2D matrix to flat 1  
#dimension vector. image size 28 x 28 = 784
#each pixel input will be input neuron
#using numpy reshape
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train.shape, x_test.shape


# In[ ]:


#pixel valus 0 to 255
#x_train.min(), x_test.max()


# In[7]:


#NN like values between 0 and 1, so we'll normalize the values 
x_train = x_train /255.0
x_test = x_test /255.0


# In[6]:


# 10 categories output
from keras.utils import to_categorical 
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes =10)


# In[2]:


from keras.layers import Input, Dense, Activation
from keras.models import Model
#define input layer
img_input = Input(shape=(784,))
#dense layers, hidden layer
x = Dense(units = 32, activation = "relu")(img_input)
#output layer
x = Dense(units = 10, activation = "softmax")(x)

model = Model(inputs = img_input, outputs = x)
model.summary()


# In[10]:


#train the model
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train, y_train, batch_size = 128, epochs = 1,validation_split = 0.2)


# In[11]:


print(model.metrics_names)
model.evaluate(x_test, y_test, batch_size = 128)


# In[12]:


preds = model.predict(x_test, batch_size = 128)


# In[13]:


preds = preds.argmax(axis =1)


# In[14]:


y_test = y_test.argmax(axis =1)


# In[15]:


preds[:10], y_train[:10]


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_test, preds))


# In[ ]:




