#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
count = 0
vidcap = cv2.VideoCapture('path to dataset video')
success,image = vidcap.read()
success = True
while success:
    success,image = vidcap.read()
    
    # save frame as JPEG file
    cv2.imwrite("path to directory of saving//xyz.jpg" % count, image) 

    # exit if Escape is hit
    if cv2.waitKey(10) == 27:                     
        break
    count += 1

