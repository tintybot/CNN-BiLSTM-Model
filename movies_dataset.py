#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data preprocessing for feeding into lstm

import cv2
import os


dataset=[]


def load_images_from_folder(folder,x):
    
    image=[]
    im=[]
    c=1
    
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        n_img=cv2.resize(img,(100,100))
        if img is not None :
            im.append(n_img)
        if c%10==0:
            image.append(im)
            image.append(x)
            im=[]
            dataset.append(image)
            image=[]
        c=c+1
        
load_images_from_folder("Datasetframes\\Movies\\training\\nonviolent",0)
load_images_from_folder("Datasetframes\\Movies\\training\\violent",1)


# In[ ]:


print(len(dataset))


# In[ ]:


import random
random.shuffle(dataset)
for samples in dataset[:10]:
    print(samples[1])
x=[]
y=[]
for img,lab in dataset:
    x.append(img)
    y.append(lab)


# In[ ]:


import numpy as np
x=np.array(x).reshape(-1,10,100,100,3)
print(x.shape)


# In[ ]:


import pickle
pickle_out=open("xmovie.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()


pickle_out=open("ymovie.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[ ]:


#loading from pickle
import pickle
pickle_in=open("x.pickle","rb")
x=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)
pickle_in.close()


# In[ ]:


#data preprocessing for feeding into lstm

import cv2
import os


validset=[]


def load_images_from_folder(folder,x):
    
    image=[]
    im=[]
    c=1
    
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        n_img=cv2.resize(img,(100,100))
        if img is not None :
            im.append(n_img)
        if c%10==0:
            image.append(im)
            image.append(x)
            im=[]
            validset.append(image)
            image=[]
        c=c+1
        
load_images_from_folder("Datasetframes\\Movies\\training\\nonviolent",0)
load_images_from_folder("Datasetframes\\Movies\\training\\violent",1)


# In[ ]:


print(len(validset))


# In[ ]:


import random
random.shuffle(validset)
for samples in validset[:10]:
    print(samples[1])
x_valid=[]
y_valid=[]
for img,lab in validset:
    x_valid.append(img)
    y_valid.append(lab)


# In[ ]:


import numpy as np
x_valid=np.array(x_valid).reshape(-1,10,100,100,3)
print(x.shape)


# In[ ]:


x_valid=x_valid.astype('float32')/255
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:


from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed,Dropout, Activation, Flatten,Conv2D, MaxPooling2D,LSTM,Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard
K.set_image_dim_ordering('tf')
import numpy as np
x=x.astype('float32')/255
NAME="MOVIEFINALUlti"
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))
## training the CNN
cnn = Sequential()
#input
cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#1st layer
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
#converting to 1-d tensor
cnn.add(Flatten())

model=Sequential()
model.add(TimeDistributed(cnn,input_shape=x.shape[1:]))
model.add(Bidirectional(LSTM(32)))
#model.add(LSTM(32))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
#model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
print(model.summary())
model.fit(x,y,epochs=25,validation_data=(x_valid,y_valid),batch_size=5,callbacks=[tensorboard])
model.save("Movie.h5")


# In[ ]:




