import random
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import getfeature
def get_files(file_dir, ratio):
    images=[]
    labels=[]
    labellist=os.listdir(file_dir)
    for label in labellist:
        for img in os.listdir('%s/%s'%(file_dir,label)):
            imgpath='%s/%s/%s'%(file_dir,label,img)
            images.append(imgpath)
            lb=labellist.index(label)
            labels.append(lb)
    onehot_encoder = OneHotEncoder(sparse=False)
    labels=np.array(labels)
    labels = labels.reshape(len(labels), 1)
    labels = onehot_encoder.fit_transform(labels)
    tra_images, tra_labels, val_images, val_labels= train_test_split(images, labels, test_size=ratio, random_state=random.randint(0, 100))
    return  tra_images, val_images, tra_labels, val_labels
def getbatch(image, label,i, batch_size):
    image_batch=[]
    label_batch=[]
    for index in range(i*batch_size , (i+1)*batch_size):
        img=getfeature.read(image[index])
        image_batch.append(img)
        label_batch.append(label[index])
    image_batch=np.array(image_batch)
    label_batch = np.array(label_batch)
    return image_batch, label_batch
def gettest(image, label):
    image_batch=[]
    label_batch=[]
    for index in range(0,len(image)):
        img=getfeature.read(image[index])
        image_batch.append(img)
        label_batch.append(label[index])
    image_batch=np.array(image_batch)
    label_batch = np.array(label_batch)
    return image_batch, label_batch