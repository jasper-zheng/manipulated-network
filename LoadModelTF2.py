# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:32:10 2022

@author: 21009460
"""
from matplotlib import pyplot as plt

from tensorflow import config

import tensorflow as tf

physical_devices = config.list_physical_devices('GPU')
print(physical_devices)

import numpy as np
from stylegan import StyleGAN_G


#%%


model = StyleGAN_G(resolution=1024)



#%%

import pickle
all_weights = pickle.load(open('gs_weights_celebahq.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model
copy_weights_to_keras_model(model.model_mapping, all_weights)
copy_weights_to_keras_model(model.model_synthesis, all_weights)



#%%
rnd = np.random.RandomState(9)
latents = rnd.randn(1, 512)

#%%
y = model.generate_sample_from_vector(latents,is_visualize=True)


#%%

print(model.layers[0].summary())
print(model.layers[1].summary())

#%%

weights_and_bias = model.layers[1].layers[20].get_weights()

weights = weights_and_bias[0]



#%%

weights_new = weights.copy()

#%%

weights_new = np.zeros(weights_new.shape)


#%%

model.layers[1].layers[20].set_weights([weights_new])





#%%


y_s = model.generate_sample_from_vector(latents,is_visualize=True)


#%%




def find_closest_latent_vector(initial_vector, 
                               num_optimization_steps,
                               steps_per_image,
                               target_image,
                               latent_dim = 512):
  images = []
  losses = []

  vector = tf.Variable(initial_vector)  
  optimizer = tf.optimizers.Adam(learning_rate=0.01)
  loss_fn = tf.losses.MeanAbsoluteError(reduction="sum")
  
  latents_vector = tf.convert_to_tensor(initial_vector)
  
  for step in range(num_optimization_steps):
    if (step % 100)==0:
      print()
    print('.', end='')
    with tf.GradientTape() as tape:
      #image = progan(vector.read_value())['default'][0]
      tape.watch(latents_vector)
      
      image = model.predict(latents_vector)[0]
      image = tf.Variable(image).read_value()
      image = tf.cast(image, dtype=tf.double)
      if (step % steps_per_image) == 0:
        images.append(image.numpy())
      target_image_difference = loss_fn(image, target_image[:,:,:3])
      # The latent vectors were sampled from a normal distribution. We can get
      # more realistic images if we regularize the length of the latent vector to 
      # the average length of vector from this distribution.
      regularizer = tf.cast(tf.abs(tf.norm(latents_vector) - np.sqrt(latent_dim)), dtype=tf.float64)

      loss = target_image_difference + regularizer
      losses.append(loss.numpy())

    # Here we update the optimized vector
    grads = tape.gradient(loss, [latents_vector])
    optimizer.apply_gradients(zip(grads, [latents_vector]))
    
  return images, losses








#%%

num_optimization_steps=200
steps_per_image=5
images, loss = find_closest_latent_vector(latents, 
                                          num_optimization_steps, 
                                          steps_per_image,
                                          target_image=y[0])


#%%

#%%

weights_and_bias_10 = model.layers[1].layers[-10].get_weights()
weights_and_bias_20 = model.layers[1].layers[50].get_weights()

weights_10 = weights_and_bias_10[0]
weights_20 = weights_and_bias_20[0]

#%%

weights_shrink = weights_10

weights_shrink[:,:,:16,:16] = np.zeros(weights_shrink[:,:,:16,:16].shape)
#%%

weights_shrink = weights_10
weights_shrink[:,:,0,0] = np.zeros(weights_10.shape)

#%%

weights_shrink = weights
weights_shrink = 0.01*weights_shrink


#%%

model.layers[1].layers[-10].set_weights([weights_shrink])




#%%


y_s = model.generate_sample(seed=6,is_visualize=True)


#%%
plt.figure(figsize=(20,20))
plt.imshow(y_s[0])


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

  return 0#interpolated_images



#%%

latent_seed = [10,11,18,20,25,28]

lat_inter = []

for i in range(len(latent_seed)-1):
    
    rnd = np.random.RandomState(latent_seed[i])
    lat_this = rnd.randn(18, 512)
    rnd = np.random.RandomState(latent_seed[i+1])
    lat_next = rnd.randn(18, 512)
    if i==0:
        lat_inter = interpolate_hypersphere(lat_this,lat_next,120)
    else:
        lat_inter = tf.concat([lat_inter,interpolate_hypersphere(lat_this,lat_next,120)],0)
    
'''
rnd = np.random.RandomState(10)
lat_10 = rnd.randn(18, 512)
rnd = np.random.RandomState(11)
lat_11 = rnd.randn(18, 512)

lat_inter = interpolate_hypersphere(lat_10,lat_11,120)
'''

#%%
y_inter_imgs = []
for i,lat in enumerate(lat_inter):
    y_inter = model.layers[1].predict(lat)
    image = y_inter.transpose([0, 2, 3, 1])
    image = np.clip((image+1)*0.5, 0, 1)
    y_inter_imgs.append(image)
    filename = f'002/latent_{i:05d}.png'
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(image[0])
    plt.savefig(filename)
    plt.close()
    
    print(i)

    

#%%


rnd = np.random.RandomState(14)
lat_this = rnd.randn(1, 512)





#%%

styles = model.layers[0].predict(lat_this)
#%%
styles[0,0:4,:] = rnd.randn(4,512)

#%%

y_inter = model.layers[1].predict(styles)
image = y_inter.transpose([0, 2, 3, 1])
image = np.clip((image+1)*0.5, 0, 1)
plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(image[0])



#%%

latent_seed = [10,18,20,25,28]

lat_inter = []
styles_this = model.layers[0].predict(lat_this)
styles_next = model.layers[0].predict(lat_this)

for i in range(len(latent_seed)-1):
    
    rnd = np.random.RandomState(latent_seed[i])
    styles_this[0,0:4,:] = rnd.randn(4,512)
    rnd = np.random.RandomState(latent_seed[i+1])
    styles_next[0,0:4,:] = rnd.randn(4,512)
    if i==0:
        lat_inter = interpolate_hypersphere(np.copy(styles_this),np.copy(styles_next),160)
    else:
        lat_inter = tf.concat([lat_inter,interpolate_hypersphere(np.copy(styles_this),np.copy(styles_next),160)],0)



#%%






#%%





