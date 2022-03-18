# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:16:55 2022

@author: 21009460
"""
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import config

physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

import numpy as np


import pickle

features = pickle.load( open( f"./feature_maps/features_32x32_7.p", "rb" ) )[0]

#%%

plt.imshow(np.mean(features,axis=0), cmap='Greys')

#%%

inputs = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.mean(features,axis=0)),axis=0),axis=0)

#%%
count = 0
fig,axs = plt.subplots(1,5,figsize=(15,3))
for i in range(-4,5,2):
    x = inputs.shape[-1]
    x_new = int(inputs.shape[-1]+i)
    print(f'operation: scale: {x} -> {x_new}')
    base_t = tf.transpose(inputs,[1,2,3,0])
    base_re = tf.image.resize(base_t, [x_new,x_new], method='gaussian')
    base_crop = tf.image.resize_with_crop_or_pad(base_re,x,x)
    out = tf.transpose(base_crop,[3,0,1,2])
    
    axs[count].title.set_text(f'scale: {i}')
    axs[count].imshow(out[0,0,:,:], cmap='Greys')
    count+=1


