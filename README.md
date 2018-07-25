# Two-Stream-CNN implement in Keras
Two Stream CNN is proposed in [__SKELETON-BASED ACTION RECOGNITION WITH CONVOLUTIONAL NEURAL NETWORKS__](https://arxiv.org/abs/1704.07595), which is used for skeleton-based action recognition. It maps a skeleton sequence to an image( coordinates x,y,z to image R,G,B ). And they specially designed skeleton transformer module to rearrange and select important skeleton joints automatically.
## Requirments
* Python3
* Keras
* h5py
* matplotlib
* numpy
## Network Architecture
The network mainly consists of four modules which are `Skeleton Transformer`, `ConvNet`, `Feature Fusion` and `Classification`. The inputs of two stream are raw data(x, y, z) and frame difference respectively. As show below :
