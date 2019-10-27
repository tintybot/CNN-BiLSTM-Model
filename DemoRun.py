#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2

modelcnnlstm=tf.keras.models.load_model("Movie.h5")
test=[]


def load_images_from_folder(folder):
    
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
            #image.append(x)
            im=[]
            test.append(image)
            image=[]
        c=c+1
        
load_images_from_folder("sampleframes")
def detection_of_violent_activities(vid_seq):
    
    #reshaping the 10 frames as per the model
    print(len(vid_seq))
    #vid_seq=vid_seq/225
    test=[]
    test.append(vid_seq)
    inputtestshape=[]
    inputtestshape.append(test)
    
    #convert the list into an array
    inputtestshape=np.array(inputtestshape).reshape(-1,10,100,100,3)
    
    #predicting the probabilities of violent and non-violent activities
    prediction=modelcnnlstm.predict_proba(inputtestshape)
    print(prediction)
    #calculation the maximum probability
    val=prediction[0][1]*100
    result=np.argmax(prediction)
    return(val)
detection_of_violent_activities(test)


# In[ ]:


print(prediction[0][result]*100)


# In[ ]:


8.6445965e-05*100


# In[ ]:


7.7089721e-01*100


# In[ ]:


import numpy as np
a=[1,2,3,4]
a=np.array(a)
print(a)
print(list(a))


# In[ ]:




