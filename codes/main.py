# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:31:06 2022

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


#%%

from stylegan import StyleGAN_G

#%%
model = StyleGAN_G(resolution=1024)

#%%

all_weights = pickle.load(open('gs_weights_ffhq.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model

copy_weights_to_keras_model(model.model_mapping, all_weights)
copy_weights_to_keras_model(model.model_synthesis, all_weights)
model.copy_weights(all_weights)

#%%

rnd = np.random.RandomState(28526)
latents = rnd.randn(1, 512)

marks_m = [32]
marks_v = [28526,852267,522634,5236,5361,536778]
#%%
y = model.generate_sample_from_vector(latents,is_visualize=True)


#%%

#labels = pickle.load( open( "labels_16x16_Conv0_vggCluster_8_updated.p", "rb" ) )
clusters = []
for i in range(21):
    if i<8:
        shape = 2**(int(i/2)+2)
        if i<2:
            cluster_type = 'affinityCluster'
        else:
            cluster_type = 'vggCluster'
        clusters.append(pickle.load( open( f"./clustered_maps/labels_{shape}x{shape}_{i}_{cluster_type}_8.p", "rb" ) ))
    else:
        clusters.append([])

#%%
operations = [{'resolution':4,
               'install_after':'Conv0_up',
               'layers':[]
               },
              {'resolution':4,
                'install_after':'Conv1',
                'layers':[]
               },
              {'resolution':8,
                'install_after':'Conv0_up',
                'layers':[]
               },
              {'resolution':8,
                'install_after':'Conv1',
                'layers':[{'operation':'sharpen',
                           'name':'01',
                           'sharpen_factor':3,
                           'with_norm':True,
                           'clusters':[1,5,6]}]
               },
              {'resolution':16,
                'install_after':'Conv0_up',
                'layers':[{'operation':'erosion',
                           'name':'01',
                           'with_norm':True,
                           'clusters':[1,5,6]},
                          
                          {'operation':'mean_filter',
                            'name':'02',
                            'with_norm':True,
                            'filter_shape':5,
                            'clusters':[3]},
                          {'operation':'mean_filter',
                            'name':'02',
                            'with_norm':True,
                            'filter_shape':3,
                            'clusters':[2,3]},
                          {'operation':'scale',
                            'name':'02',
                            'with_norm':True,
                            'scale':-6,
                            'clusters':[3,7]},
                          {'operation':'scale',
                           'name':'02',
                           'scale':8,
                           'with_norm':True,
                           'clusters':[1,5,6]},
                          {'operation':'erosion',
                           'name':'03',
                           'with_norm':True,
                           'clusters':[1,5,6]},
                          {'operation':'erosion',
                           'name':'03',
                           'with_norm':True,
                           'clusters':[1,5,6]},
                          {'operation':'erosion',
                           'name':'03',
                           'with_norm':True,
                           'clusters':[4]},
                          {'operation':'dilation',
                            'name':'02',
                            'with_norm':True,
                            'clusters':[3,7]},
                          {'operation':'scale',
                           'name':'02',
                           'scale':-4,
                           'with_norm':True,
                           'clusters':[3,7]},
                          {'operation':'scale',
                           'name':'02',
                           'scale':-8,
                           'with_norm':True,
                           'clusters':[0]},
                          {'operation':'rotate',
                           'name':'02',
                           'rotation':np.pi/1.3,
                           'with_norm':True,
                           'clusters':[]}
                          ]
               },
              {'resolution':16,
                'install_after':'Conv1',
                'layers':[]
               },
              {'resolution':32,
                'install_after':'Conv0_up',
                'layers':[]
               },
              {'resolution':32,
                'install_after':'Conv1',
                'layers':[]
               },
              ]
#%%

operations = [{'resolution':4,
               'install_after':'Conv0_up',
               'layers':[]
               },
              {'resolution':4,
                'install_after':'Conv1',
                'layers':[]
               },
              {'resolution':8,
                'install_after':'Conv0_up',
                'layers':[]
               },
              {'resolution':8,
                'install_after':'Conv1',
                'layers':[]
               },
              {'resolution':16,
                'install_after':'Conv0_up',
                'layers':[{'operation':'sin_disrupt',
                           'name':'21',
                           'sinX':3.35,
                           'sinY':3.35,
                           'translateX':0,
                           'translateY':0,
                           'with_norm':True,
                           'clusters':[]}
                          ]
               },
              {'resolution':16,
                'install_after':'Conv1',
                'layers':[{'operation':'sharpen',
                          'name':'21',
                          'sharpen_factor':np.random.random()*0.4,
                          'with_norm':True,
                          'clusters':[]}]
               },
              {'resolution':32,
                'install_after':'Conv0_up',
                'layers':[]
               },
              {'resolution':32,
                'install_after':'Conv1',
                'layers':[]
               },
              ]

#%%




#%%

model.rebuild_operations(clusters, operations)
model.generate_from_vector_fast(latents)


#%%

rnd = np.random.RandomState(28526)
latents = rnd.randn(1, 512)

rnd = np.random.RandomState(536778)
latents_2 = rnd.randn(1, 512)

marks_m = [32]
marks_v = [28526,852267,522634,5236,5361,536778]
#%%
model.generate_from_vector_fast(latents)
#y = model.generate_sample_from_vector(latents,is_visualize=True)

#%%
model.generate_from_vector_fast(latents)

#%%

y = model.model_mapping(latents)

y = model.model_synthesis.predict(y)

#%%

show_raw(y)

#%%

def show_raw(y,title=''):
    images = y.transpose([0, 2, 3, 1])
    images = np.clip((images+1)*0.5, 0, 1)
    plt.figure(figsize=(21,21))
    plt.axis('off')
    plt.title(title)
    plt.imshow(images[0])
    plt.show()

#%%

from numpy.random import randn
from numpy import linspace, asarray

 # generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return asarray(vectors)

#%%

l_1 = generate_latent_points(512,)
#%%


f16_0 = pickle.load( open( f"./feature_maps/features_16x16_4.p", "rb" ) )[0]


#%%
from tensorflow.keras.backend import clear_session
from numba import cuda


def produce_base(i,dx,dy):
    w = 16
    h = 16
    base = np.zeros((1,512,h,w))
    for x in range(w):
        for y in range(h):
            base[:,:,x,y] += (np.sin((x-4-dy)/w*3.35))
            base[:,:,x,y] += (np.sin((y-2-dx)/w*3.35))
    out = tf.convert_to_tensor(base,dtype=tf.float32)
    plt.axis('off')
    plt.imshow(out[0][0],cmap='Greys')
    plt.savefig(f'./test_animation/base_{i:04}.png')
    plt.close()

def swap_operation(operations,x,y):
    operations[4]['layers'][0]['translateX'] = x
    operations[4]['layers'][0]['translateY'] = y
    return operations
#%%
x = 0
y=0
speedX=0.05
speedY=0.08

for i in range(1000):
    x -= speedX
    y -= speedY
    speedY -= 0.002
    if x <-5 or x >4:
        speedX *= -1
    if y <-5 or y >4:
        speedY *= -1
    
    #operations = swap_operation(operations,x,y)

    #model.rebuild_operations(clusters, operations)
    #img = model.generate_from_inter(all_latents[i],is_visualize=False)
    
    img = produce_base(i,x,y)
    
    #plt.figure(figsize=(21,21))
   # plt.axis('off')
    #plt.imshow(img)
   # plt.savefig(f'./003/g_{i:04}.png')
    #plt.close()
    
    
    #clear_session()

    
#%%

inter_latent_1 = model.layers[0].predict(latents)
inter_latent_2 = model.layers[0].predict(latents_2) 
    
#%%
model.generate_from_inter(inter_latent_2)


#%%

inter_latent_2[:,5:,:] = inter_latent_1[:,5:,:]


#%%
def interpolate_hypersphere(v1, v2, num_steps):
  v1_norm = tf.norm(v1)
  v2_norm = tf.norm(v2)
  v2_normalized = v2 * (v1_norm / v2_norm)

  vectors = []
  for step in range(num_steps):
    interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
    interpolated_norm = tf.norm(interpolated)
    interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
    vectors.append(interpolated_normalized)
  return tf.stack(vectors)

def interpolate_between_vectors(_v1,_v2,n_step):
  v1 = _v1 
  v2 = _v2
    
  # Creates a tensor with 25 steps of interpolation between v1 and v2.
  vectors = interpolate_hypersphere(v1, v2, n_step)
  
  # Uses module to generate images from the latent space.
  #interpolated_images = progan(vectors)['default']

  return 0 #interpolated_images
#%%

all_latents = interpolate_hypersphere(inter_latent_1,inter_latent_2,1000)

