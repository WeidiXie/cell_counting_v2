# cell_counting_v2

The repository includes the code for training cell counting applications.

Related Papers:

[1] Microscopy Cell Counting with Fully Convolutional Regression Networks.

http://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf

[2] U-Net: Convolutional Networks for Biomedical Image Segmentation.

https://arxiv.org/abs/1505.04597

I will continue adding more network architectures.

To make the training easier, I added Batch Normalization to all architectures.

In all architectures, they follow the fully convolutional idea, each architecture consists of a down-sampling path,
followed by an up-sampling path. 
During the first several layers, the structure resembles the cannonical classification CNN, as convolution,
ReLU, and max pooling are repeatedly applied to the input image and feature maps. 
In the second half of the architecture, spatial resolution is recovered by performing up-sampling, convolution, eventually mapping the intermediate feature representation back to the original resolution. 

In the U-net version, low-level feature representations are fused during upsampling, aiming to compensate the information loss due to max pooling.

As people know, Deep Learning is developing extremely fast today, both papers were published two years ago,
which is quite "old". If people are interested in cell counting, please edit based on this.




