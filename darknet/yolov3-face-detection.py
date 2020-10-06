#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import darknet
import numpy as np
import cv2
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os


# In[3]:


def save_boxImage(imagePath, boxes, path):
    cfg = './cfg/yolov3.cfg'
    weights = './cfg/yolov3.weights'
    data = './cfg/coco.data'
    
#     img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    img = cv2.imread(imagePath)
    img_cp = img.copy()
    
    for b in boxes:
        box = b[2]
        if b[1] > 0.8:
            center = (int(box[0]), int(box[1]))
            width = int(box[2])
            height = int(box[3])

            top_left = (center[0]-(width//2), center[1]-(height//2))
            btm_right = (center[0]+(width//2), center[1]+(height//2))

        #     top_left = (int(box[0]), int(box[1]))
        #     btm_right = (int(box[0]+box[2]), int(box[1]+box[3]))
            img = cv2.rectangle(img,top_left,btm_right,(0,255,0),5)
            
        cv2.imwrite(path, img)
#     plt.subplot(1, 2, 1)
#     plt.imshow(img_cp)
#     plt.subplot(1, 2, 2)
#     plt.imshow(img)
    


# In[4]:


# test_dir = './cfg/temp0922/kari/dataset_detection/images/'
test_dir = '../temp1004/tmp2imgColor128/testB/'
est_dir = '../temp1004/pix2pix_results/tmp2imgC128_pix2pix/test_latest/images/'
test_list = [f.split('.jpg')[0] for f in os.listdir(test_dir) if f.endswith('.jpg')]
test_list2 = [f for f in os.listdir(est_dir) if f. endswith('_fake_B.png') and not  f. endswith('checkpoint_fake_B.png') ]

print(len(test_list), test_list[0])
print(len(test_list2), test_list2[0])


# In[5]:


# In[6]:


(1, 2) + (3, 4)


# In[7]:


out_dir = './results/img128_person_detection/'
out_dir2 = './results/img128_fake_person_detection/'
csv_dir = './csv/'

cfg = './cfg/yolov3.cfg'
weights = './cfg/yolov3.weights'
data = './cfg/coco.data'


# In[10]:


box_df = pd.DataFrame(columns=['filename', 'class', 'confedence', 'bbox'])
box_df2 = pd.DataFrame(columns=['filename', 'class', 'confedence', 'bbox'])
i = 0
for test in test_list:
    imagePath = test_dir + test + '.jpg'
    boxes = darknet.detect_box(cfg.encode(), weights.encode(), data.encode(), imagePath.encode())
    # print(boxes)
    boxes2 = [b for b in boxes if b[0].decode() == 'person']
#     print(i, test, ':', boxes2)
    
    
    
    imagePath = est_dir + test + '_fake_B.png'
    boxes = darknet.detect_box(cfg.encode(), weights.encode(), data.encode(), imagePath.encode())
    boxes3 = [b for b in boxes if b[0].decode() == 'person']
    
    
    print(i, test, ':', boxes2, boxes3)
    i += 1
#     save_boxImage(imagePath, boxes2, outPath)

    df = pd.DataFrame(boxes2, columns = [ 'class', 'confedence', 'bbox'])
    df['filename'] = test
    
    df2 = pd.DataFrame(boxes3, columns = [ 'class', 'confedence', 'bbox'])
    df2['filename'] = test
    
    box_df = pd.concat([box_df, df])
    box_df2 = pd.concat([box_df2, df2])
# box_df.head()


# In[11]:


# box_df2.head()


# In[ ]:


box_df.to_csv(csv_dir + 'pix2pix_real1004.csv')
box_df2.to_csv(csv_dir + 'pix2pix_fake1004.csv')


# In[ ]:




