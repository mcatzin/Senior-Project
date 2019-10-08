#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras 
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train.shape, x_test.shape


# In[5]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[6]:


from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape, y_test.shape


# In[14]:


from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten
from keras.models import Model

input_layer = Input(shape = (28,28,1))
_ = Conv2D(filters =32, kernel_size =(3,3))(input_layer)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)

_ = Conv2D(filters =64, kernel_size =(3,3))(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)

_ = Conv2D(filters =64, kernel_size =(3,3))(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)

_ = Flatten()(_)
_ = Dense(units=60)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ =Dense(units=10)(_)
_ = Activation("softmax")(_)

model = Model(inputs=input_layer, outputs = _)
model.summary()
#Conv Layer
#batch normalization layer
#max pooling


# In[17]:


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, batch_size=256, epochs = 6)


# In[ ]:




