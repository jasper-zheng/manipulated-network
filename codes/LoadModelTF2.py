# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:32:10 2022

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

from stylegan_aux import StyleGAN_G

#%%


#model = StyleGAN_G(resolution=1024,block_sequence=seq)
model = StyleGAN_G(resolution=1024)


#%%


all_weights = pickle.load(open('gs_weights_ffhq.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model
copy_weights_to_keras_model(model.model_mapping, all_weights)
copy_weights_to_keras_model(model.model_synthesis, all_weights)

#%%

y = model.generate_sample_from_vector(latents,is_visualize=True)

#%%


model_m = StyleGAN_G(resolution=1024)



#%%


all_weights = pickle.load(open('gs_weights_celebahq.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model
copy_weights_to_keras_model(model_m.model_mapping, all_weights)
copy_weights_to_keras_model(model_m.model_synthesis, all_weights)


#%%
rnd = np.random.RandomState(536778)
latents = rnd.randn(1, 512)

marks_m = [32]
marks_v = [28526,852267,522634,5236,5361,536778]

#%%
y = model.generate_sample_from_vector(latents,is_visualize=True)


#%%

print(model_v.layers[0].summary())
print(model_v.layers[1].summary())

#%%

weights_and_bias = model_v.layers[1].layers[20].get_weights()

weights = weights_and_bias[0]

#%%

#%%

#inter_latent_m = model_m.layers[0].predict(latents)
inter_latent_2 = model.layers[0].predict(latents)

#%%

def swap_lat(lat1,lat2,s,e):
    swap1 = np.copy(lat1)
    swap2 = np.copy(lat2)
    lat1[:,s:e,:] = swap2[:,s:e,:]
    lat2[:,s:e,:] = swap1[:,s:e,:]
    return lat1-lat2, lat2-lat1

#%%

inter_lat_m, inter_lat_v = swap_lat(inter_latent_m, inter_latent_v, 0, 18)


#%%

inter_latent_2[:,5:,:] = inter_latent[:,5:,:]

#%%

y_2 = model.layers[1].predict(inter_latent)
show_raw(y_2)

#%%

def swap_weights(model1,lat1,model2,lat2,start,end,s,e):
    for i,layer in enumerate(model1.layers[1].layers):
        if i > start:
            if i >= end:
                break
            trainables = layer.trainable_weights
            if len(trainables)>0:
                swap1 = model1.layers[1].layers[i].get_weights()
                swap2 = model2.layers[1].layers[i].get_weights()
                model1.layers[1].layers[i].set_weights(swap2)
                model2.layers[1].layers[i].set_weights(swap1)
            if i%10 == 0:
                swap_lat_1 = np.copy(lat1[:,int(i/10),:])
                swap_lat_2 = np.copy(lat2[:,int(i/10),:])
                lat1[:,int(i/10),:] = swap_lat_2
                lat2[:,int(i/10),:] = swap_lat_1
    for i,layer in enumerate(model1.layers[0].layers):
        if len(layer.trainable_weights)>0:
            if i > s:
                if i >= e:
                    break
                swap1 = model1.layers[0].layers[i].get_weights()
                swap2 = model2.layers[0].layers[i].get_weights()
                model1.layers[0].layers[i].set_weights(swap2)
                model2.layers[0].layers[i].set_weights(swap1)
            
    return lat1, lat2


#%%

def sub_weights(model1,lat1,model2,lat2,start,end):
    for i,layer in enumerate(model1.layers[1].layers):
        if i > start:
            if i >= end:
                break
            trainables = layer.trainable_weights
            if len(trainables)>0:
                
                if len(model1.layers[1].layers[i].get_weights())==1:
                    swap1 = model1.layers[1].layers[i].get_weights()[0]
                    swap2 = model2.layers[1].layers[i].get_weights()[0]
                    model1.layers[1].layers[i].set_weights([swap2-swap1])
                    model2.layers[1].layers[i].set_weights([swap1-swap2])
            if i%10 == 0:
                swap_lat_1 = np.copy(lat1[:,int(i/10),:])
                swap_lat_2 = np.copy(lat2[:,int(i/10),:])
                lat1[:,int(i/10),:] = swap_lat_2
                lat2[:,int(i/10),:] = swap_lat_1
                
    
    return lat1, lat2

#%%


inter_latent_v_swapped, inter_latent_m_swapped = sub_weights(model_m,inter_latent_v,
                                                              model_v,inter_latent_m,
                                                              0,180)



#%%

y_v = model_v.layers[1].predict(inter_lat_m)
y_m = model_m.layers[1].predict(inter_lat_v)


#%%

def show_raw(y,title=''):
    images = y.transpose([0, 2, 3, 1])
    images = np.clip((images+1)*0.5, 0, 1)
    plt.figure(figsize=(14,14))
    plt.title(title)
    plt.imshow(images[0])
    plt.show()


def show_grid(images,shape=(8,8),figsize=(20,20)):
    
    fig, axs = plt.subplots(shape[0],shape[1], figsize=figsize)
    x = shape[0]
    for i,img in enumerate(images[0]):
        img = img / np.linalg.norm(img)
        img = np.clip((img+1)*0.5, 0, 1)
        axs[int(i/x),i%x].axis('off')
        axs[int(i/x),i%x].title.set_text(f'index {i:03}')
        axs[int(i/x),i%x].imshow(img,cmap='Greys')
    
def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalized_input - 1

#%%

weights_new = weights.copy()

#%%

weights_new = np.zeros(weights_new.shape)


#%%

model_v.layers[1].layers[20].set_weights([weights_new])


#%%

for i,layer in enumerate(model_v.layers[1].layers):
    trainables = layer.trainable_weights
    if len(trainables)>0:
        print(f'{i:03} {layer.name} \t\t\tshape:{len(layer.get_weights())}')
    '''
    if layer.name[-5:]=='Conv1' or layer.name[-8:]=='Conv0_up' or layer.name[-4:]=='Conv':
        weights = layer.get_weights()[0].shape
        print(f'{i:03} {layer.name} \t\t\tshape:{weights}')
'''

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

weights_and_bias = model.layers[1].layers[30].get_weights()

weights = weights_and_bias[0]

#%%

weights_shrink = weights

weights_shrink = -weights_shrink
#%%

weights_shrink = weights_10
weights_shrink[:,:,0,0] = np.zeros(weights_10.shape)

#%%

weights_shrink = weights
weights_shrink = 0.01*weights_shrink


#%%

model_v.layers[1].layers[-10].set_weights([weights_shrink])




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

  return 0 #interpolated_images



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
    y_inter = model_v.layers[1].predict(lat)
    image = y_inter.transpose([0, 2, 3, 1])
    image = np.clip((image+1)*0.5, 0, 1)
    y_inter_imgs.append(image)
    filename = f'003_v/latent_{i:05d}.png'
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(image[0])
    plt.savefig(filename)
    plt.close()
    
    print(i)

#%%
y_inter_imgs = []
for i,lat in enumerate(lat_inter):
    y_inter = model_m.layers[1].predict(lat)
    image = y_inter.transpose([0, 2, 3, 1])
    image = np.clip((image+1)*0.5, 0, 1)
    y_inter_imgs.append(image)
    filename = f'003_m/latent_{i:05d}.png'
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(image[0])
    plt.savefig(filename)
    plt.close()
    
    print(i)

#%%


rnd = np.random.RandomState(57673)
lat_this = rnd.randn(1, 512)


#%%

layers_i = [i for i,layer in enumerate(model.layers[1].layers) if layer.name[-5:]=='Conv1' or layer.name[-8:]=='Conv0_up' or layer.name[-4:]=='Conv' or layer.name[-4:]=='bias']


#%%

def swap_weights2(model1,model2):
    for i in layers_i[7:14]:
        swap1 = model1.layers[1].layers[i].get_weights()
        swap2 = model2.layers[1].layers[i].get_weights()
        model1.layers[1].layers[i].set_weights(swap2)
        model2.layers[1].layers[i].set_weights(swap1)
        
#%%

swap_weights2(model_v,model_m)

#%%

y = model_v.generate_sample(seed=1,is_visualize=True)


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

latent_seed = [99,3,6,8538,1538,15238,4477]

lat_inter = []
styles_this = model_v.layers[0].predict(lat_this)
styles_next = model_v.layers[0].predict(lat_this)

for i in range(len(latent_seed)-1):
    
    rnd = np.random.RandomState(latent_seed[i])
    styles_this[0,0:9,:] = rnd.randn(9,512)
    rnd = np.random.RandomState(latent_seed[i+1])
    styles_next[0,0:9,:] = rnd.randn(9,512)
    if i==0:
        lat_inter = interpolate_hypersphere(np.copy(styles_this),np.copy(styles_next),240)
    else:
        lat_inter = tf.concat([lat_inter,interpolate_hypersphere(np.copy(styles_this),np.copy(styles_next),240)],0)



#%%


layers_i = [[i,layer.name,layer.output.shape.as_list()] 
            for i,layer in enumerate(model.layers[1].layers) 
            if layer.name[-5:]=='Conv1' 
            or layer.name[-8:]=='Conv0_up' 
            or layer.name[-4:]=='Conv' 
            or layer.name[-5:]=='Const']

#%%

all_weights = model_v.layers[1].layers[23].get_weights()[0]

model_v.layers[1].layers[23].set_weights([-all_weights])

#%%


y = model_v.generate_sample_from_vector(latents,is_visualize=True)

#%%
y = model_v.generate_sample(seed=7,is_visualize=True)

#%%


weights = model_v.layers[1].layers[30].get_weights()[0]



weights_t = weights.transpose((1,0,2,3))


model_v.layers[1].layers[30].set_weights([weights_t])


#%%


weights = model_v.layers[1].layers[30].get_weights()[0]

weights_fl = np.flip(weights,0)

model_v.layers[1].layers[30].set_weights([weights_fl])


#%%


weights = model_v.layers[1].layers[90].get_weights()[0]

weights_in = weights

weights_in *= -1

model_v.layers[1].layers[90].set_weights([weights_in])

#%%
from tensorflow.keras import Model

#%%
inp = model.layers[1].input
outp = model.layers[1].layers[99].output

aux = Model(inputs=inp,outputs=outp)
y = aux.predict(inter_latent_v)

#%%
show_grid(y)

#%%

def show_grid(images,shape=(12,12),figsize=(20,20)):
    
    fig, axs = plt.subplots(shape[0],shape[1], figsize=figsize)
    x = shape[0]
    for i,img in enumerate(images):
        #img = img / np.linalg.norm(img)
        #amax = np.amax(img)
        #amin = np.amin(img)
        #img = np.clip((img-amin)/(amax-amin), 0, 1)
        axs[int(i/x),i%x].axis('off')
        axs[int(i/x),i%x].title.set_text(f'index {i:03}')
        axs[int(i/x),i%x].imshow(img,cmap='Greys')
    

#%%
rnd = np.random.RandomState(28526)
latents = rnd.randn(1, 512)

ys = []
ys.append(model.generate_sample_from_vector(latents,is_visualize=False)[0])

for i in range(len(layers_i)):
    if i==0:
        print(f'manipulating layer {layers_i[i][0]:03}')
        model.layers[1].layers[layers_i[i][0]].set_weights([np.zeros(all_weights[i].shape)])
    if i>0:
        print(f'reverting layer {layers_i[i-1][0]:03}')
        model.layers[1].layers[layers_i[i-1][0]].set_weights([np.copy(all_weights[i-1])])
        print(f'manipulating layer {layers_i[i][0]:03}')
        model.layers[1].layers[layers_i[i][0]].set_weights([np.zeros(all_weights[i].shape)])
        print(' ')
    ys.append(model.generate_sample_from_vector(latents,is_visualize=False)[0])



#%%

inter_output = model.layers[1].layers[53].output


#%%

from tensorflow.keras.layers import BatchNormalization

#%%

inter_test = BatchNormalization()(inter_output)


#%%

model.layers[1].layers[54](inter_test)



#%%
from tensorflow.keras import Model
#%%

inter_latent = model.layers[0].predict(latents)

#27:8x8
#47:16x16
#67:32
size = 16



inpt = model.layers[1].input
oupt = model.layers[1].layers[53].output

model_aux = Model(inputs=inpt,outputs=oupt)

features = model_aux.predict(inter_latent)

#%%
size_and_i = [[7,4],[15,4],
              [24,8],[33,8],
              [44,16],[53,16],
              [64,32],[73,32]]

inter_latent = model.layers[0].predict(latents)

for i,l in enumerate(size_and_i):
    size = l[1]
    inpt = model.layers[1].input
    oupt = model.layers[1].layers[l[0]].output
    
    model_aux = Model(inputs=inpt,outputs=oupt)
    features = model_aux.predict(inter_latent)
    
    pickle.dump( features, open( f"./feature_maps/features_{l[1]}x{l[1]}_{i}.p", "wb" ) )

#%%
pickle.dump( features, open( "features_16x16_Conv1.p", "wb" ) )

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#%%

features_rs = features[0].reshape((512,size*size))

scaler = StandardScaler()
features_std = scaler.fit_transform(features_rs).reshape(512,size,size)

#%%

kmeans = KMeans(n_clusters = 32, random_state = 0)
clusters = kmeans.fit_predict(features_std)
print(kmeans.cluster_centers_.shape)



#%%

centers = kmeans.cluster_centers_.reshape(32, size, size)
labels = kmeans.labels_

fig, ax = plt.subplots(8, 4, figsize = (8,16))
for i, (axi, center) in enumerate(zip(ax.flat, centers)):
    axi.set(xticks = [], yticks = [])
    axi.title.set_text(i)
    axi.imshow(center, interpolation = 'nearest', cmap = plt.cm.binary)
    
    
#%%

# set to 1 to apply the operation on the selected layers

#marked_clusters = [2,3,7,14,16,28,31]
#marked_clusters = [6,8,10]
marked_clusters = [1,5,6] # eye mouse nose
#marked_clusters = [1]
#marked_clusters = [1,4,7]

seq = np.array( [1 if l in marked_clusters else 0 for l in labels] )
#seq = np.array( [1 if True else 0 for l in labels] )

#%%

labels = pickle.load( open( "labels_16x16_Conv0_vggCluster_8_updated.p", "rb" ) )
clusters = []
for i in range(21):
    if i == 4 or i==3:
        clusters.append(labels)
    else:
        clusters.append([])

#%%

from stylegan import StyleGAN_G


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

#clusters = []




#%%


model = StyleGAN_G(resolution=1024,clusters=clusters,operations=operations)

#%%
all_weights = pickle.load(open('gs_weights_ffhq.pkl', 'rb'))
print('Weights loaded to memory.')

from stylegan import copy_weights_to_keras_model

copy_weights_to_keras_model(model.model_mapping, all_weights)
copy_weights_to_keras_model(model.model_synthesis, all_weights)


#%%

y = model.generate_sample_from_vector(latents,is_visualize=True)

#%%

rnd = np.random.RandomState(28526)
latents = rnd.randn(1, 512)

marks_m = [32]
marks_v = [28526,852267,522634,5236,5361,536778]

y = model.generate_sample_from_vector(latents,is_visualize=True)

#%%



#%%

plt.figure(figsize=(21,21))
plt.axis('off')
plt.imshow(y[0])

#%%
pickle.dump( operations, open( "002.p", "wb" ) )
#%%



all_weights = model.layers[1].layers[67].get_weights()[0]
all_weights_tf = tf.convert_to_tensor(all_weights)



#%%
inputs = tf.transpose(all_weights_tf,[3,2,0,1])
#%%

ones = tf.ones(tf.shape(inputs))
diag = tf.cast(tf.linalg.diag(seq), tf.float32)
#inputs += tf.ones(tf.shape(inputs))
scaled = tf.tensordot(diag,tf.transpose(ones,[1,2,3,0]),axes=1)
#inputs -= tf.ones(tf.shape(inputs))
mask = tf.cast(tf.transpose(scaled,[3,0,1,2]),dtype=tf.bool)
print(mask.shape)


pattern = np.where(seq==1)

based_t = tf.transpose(inputs,[1,2,3,0])
based_sliced = tf.gather(based_t,pattern[0])

lens = tf.shape(based_sliced)[0]
print(lens)

based_repeat = tf.repeat(based_sliced,int(512/lens),axis=0)
print(f'sliced {based_sliced.shape}')
based_full = tf.concat([based_repeat,based_sliced[:512%lens]], axis=0)
out_filled = tf.transpose(based_full,[3,0,1,2])

out = tf.where(mask,inputs,out_filled)

#%%

out_weights = tf.transpose(out,[2,3,1,0]).numpy()

#%%
model.layers[1].layers[48].set_weights([out_weights])














