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
![Two Stream CNN](/layers/network.png)
## Usage
1. __function/data_generator.py__ : generate the inputs numpy array of two stream  

2. __layers/transformer__ : the layer of Skeleton Transformer implement in Keras  

3. __network/__ : the fold has four flies with different feature fusion way
## Result
|  model  |  accuracy(cs)  |
| :---------: | :---------: |
|  base line  | 83.2% |
|  my model   | 80.7% |  

Introduce `attention mechanism` to Skeleton Transformer module. Then, the accurancy can reach at 82.1%.
## Contact
If you have any questions, please feel free to contact me.  
Duohan Liang (duohanl@outlook.com)
