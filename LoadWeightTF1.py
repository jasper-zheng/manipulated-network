# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:43:23 2022

@author: 21009460
"""

import numpy as np
import collections
import tensorflow as tf
import pickle
import dnnlib.tflib as tflib

#%%


tflib.init_tf()
f = open('karras2019stylegan-cats-256x256.pkl', 'rb')
_G, _D, _Gs = pickle.load(f)
print('Gs ready')
#%%

# Load best weights (Gs) into memory
sess = tf.get_default_session()
all_weights = collections.OrderedDict()
with sess.as_default():  
    for i, (key, weight_tensor) in enumerate(_Gs.vars.items()):
        all_weights[key] = weight_tensor.eval()
        print('.', end='')
print(' ({}) weights found. '.format(len(all_weights)))

#%%

pickle.dump(all_weights, open( 'gs_weights_cats.pkl', 'wb' ) )
print('Saved original StyleGAN weights to disk.')


#%%
