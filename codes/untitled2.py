# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:32:10 2022

@author: 21009460
"""
from matplotlib import pyplot as plt

import numpy as np
from stylegan import StyleGAN_G

model = StyleGAN_G()


#%%

import pickle
all_weights = pickle.load(open('gs_weights.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model
copy_weights_to_keras_model(model.model_mapping, all_weights)
copy_weights_to_keras_model(model.model_synthesis, all_weights)



#%%

y = model.generate_sample(is_visualize=True)


#%%


#%%

#%%