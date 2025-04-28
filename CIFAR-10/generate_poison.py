# # Threat Model
#
# This script loads the CIFAR-10 dataset and perform a random poisoning attack on the dataset.
# It then saves the poisoned data, the trigger (for deployment attacks), and the source and target classes.

from utils import data_loader_CIFAR10
from utils.backdoor_attack import generate_poisoned_data
from skimage.color import rgb2hsv,hsv2rgb
from skimage.exposure import equalize_hist
from skimage.io import imread
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

# ### Stretch Value
# This function will equalizes the 'value' in HSV for images

ATTACK_NAME = "Badnet"

def stretch_value(img):
    flag=False
    if img.max()<=1.:
        img=(255*img).astype('uint8')
        flag=True
    img=rgb2hsv(img)
    img[...,2]=equalize_hist(img[...,2])
    if flag:
        return (hsv2rgb(img)*255).astype('uint8')
    else:
        return hsv2rgb(img)

# load data and apply stretch value to each image (takes a while!)

X_train, y_train, X_val, y_val, X_test, y_test = data_loader_CIFAR10.load_CIFAR10()
X_train=np.stack([stretch_value(img) for img in X_train],0)
X_val=np.stack([stretch_value(img) for img in X_val],0)
X_test=np.stack([stretch_value(img) for img in X_test],0)


mask_list = glob.glob("./Data/Masks/*")

# Generate attacks for every pair of source and targets
attacked_data_folder = f'./Attacked_Data/{ATTACK_NAME}/test'
if not os.path.isdir(attacked_data_folder):
    os.makedirs(attacked_data_folder)

count=0
labels=np.arange(10)

for source in range(10):
    target_labels=np.concatenate([labels[:source],labels[source+1:]])
    for target in target_labels:
        # Save the attacked data
        for k in range(10, 20):  # for testing
            trigger = imread(mask_list[k])
            X_poisoned, Y_poisoned, trigger, ind = generate_poisoned_data(X_train, y_train, source, target, trigger)
            X_poisoned_val, Y_poisoned_val, trigger_val, ind_val = generate_poisoned_data(X_val, y_val, source, target, trigger)
            with open(attacked_data_folder + '/backdoor%04d.pkl' % count, 'wb') as f:
                pickle.dump([X_poisoned, Y_poisoned, trigger, source, target], f)
            with open(attacked_data_folder + '/backdoor%04d_test.pkl' % count, 'wb') as f:
                pickle.dump([X_poisoned_val, Y_poisoned_val, trigger_val, source, target], f)

            
            count += 1

# Save the value stretched images

f=open('./Data/CIFAR10/train_heq.p','wb')
pickle.dump([X_train,y_train],f)

f=open('./Data/CIFAR10/test_heq.p','wb')
pickle.dump([X_test,y_test],f)

f=open('./Data/CIFAR10/val_heq.p','wb')
pickle.dump([X_val,y_val],f)


