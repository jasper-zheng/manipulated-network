# Manipulated Network and Feature Map Clustering  
###### Msc Artificial Intelligence for Media     
Jasper Zheng (Shuoyang) / 21009460  

![cover](./git_graphics/01.png)   

This project explores three methods of manipulating pre-trained StyleGAN [1] models: intermediate latent space truncation [2], layers' weights manipulation and Network Bending [3]. It also used VGG16 feature extraction model and KMean algorithm to cluster the feature maps in the intermediate layers to create more interpretable outcomes. Finally, it re-implemented a set of network bending operations to a code interface and showcased a series of novel images produced by the manipulated models.  

[Project Report](./project_report.pdf)

## Implementation    

#### Requirements  
The code explicitly require `python 3.7`, `tensorflow==2.3.0`, `tensorflow-addons==0.13.0`, `numpy==1.19.0`.  


#### Intermediate Latent Space Truncation  

StyleGAN generator has a mapping network and a synthesis network. The mapping network re-iterates the base latent vector with shape (1,512) and outputs an intermediate latent vector with shape (1,18,512). I separated these two networks and intercepted the intermediate vector. Then add deviations to different levels to create new variations to the generated image.  
