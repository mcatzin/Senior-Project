#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2


# In[2]:


train_path = 'C:/Users/mcatz/OneDrive/Documents/train'
valid_path = 'C:/Users/mcatz/OneDrive/Documents/valid'
test_path = 'C:/Users/mcatz/OneDrive/Documents/test'


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),batch_size=4)


# In[4]:


model = Sequential([
    Dense(16,activation ='relu', input_shape=(224,224,3)),
     Flatten(),
    Dense(32, activation ='relu'),
    Dense(20, activation='softmax')
])


# In[5]:


model.summary();


# In[6]:


model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[7]:


model.fit_generator(train_batches, steps_per_epoch=23,
                   validation_data=valid_batches, validation_steps=4, epochs=20, verbose=2)


# # Prediction

# In[27]:


predictions = model.predict_generator(test_batches, steps=10, verbose=0)


# In[28]:


predictions


# # Code block to plot the Confusion Matrix

# In[22]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,ha="right")
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, cm[i,j],
                horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    


# In[24]:


test_labels = test_batches.classes


# In[25]:


test_labels


# # Confusion Matrix

# In[29]:


cm = confusion_matrix(test_labels, predictions.argmax(axis=1))


# In[30]:


cm_plot_labels = ['Calculator','Glasses','Pencils','Phones','Shoes','Trash Can','backpack','ball','book','chair','flashlight','flower','hat','headphones'
                 ,'laptop','mouse','ring','table','tablet','watches']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

