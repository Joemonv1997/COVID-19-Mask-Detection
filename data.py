import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
data=[]
target=[]
path='D://JMASKDETECTOR/data'
listp=os.listdir(path)
categories=[i for i in range (len(listp))]
label_dict=dict(zip(listp,categories))
img_size=100
print(label_dict)
print(categories)
print(listp)

for category in listp:
    imgpath=os.path.join(path,category)
    imglist=os.listdir(imgpath)
    for imgname in imglist:
        imgf=os.path.join(imgpath,imgname)
        img=cv2.imread(imgf)
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            RI=cv2.resize(gray,(img_size,img_size))
            data.append(RI)
            target.append(label_dict[category])
        except Exception as e:
            print(e)


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)
new_target=to_categorical(target)

np.save('data',data)
np.save('target',new_target)
