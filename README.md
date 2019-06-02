# keras-triplet-center-loss
A simple Keras implementation of Triplet-Center Loss on the MNIST dataset. 
As a reference in this repository also implementations of other two similar losses, 
Center-Loss and Triplet-Loss are included. 

The Center-Loss implementation is from 

The Triplet-Loss implementation is from 

------
### Triplet-Center Loss

Triplet-Center Loss has been introduced by He et al. in https://arxiv.org/abs/1803.06189. 
It is an "hybrid" loss between Center Loss and Triplet Loss that allows to maximise inter-class distance and 
minimize intra-class distance.


### Details 
In this repository a simple implementation on the MNSIT or alternatively Fashion MNIST is 
shown. 

Running **main.py** will start sequentially 4 training routines with 4 different losses:

* Categorical Crossentropy only 
* Center-loss + Categorical Crossentropy 
* Triplet-loss + Categorical Crossentropy 
* Triplet-Center loss + Categorical Crossentropy 

In Folder **runs** there will be the results of those models, including Tensorboard summaries. 
Also T-SNE is run on the embeddings to visualize how the network internal representation changes as the loss is changed.

![Image of Triplet_Center_Loss](https://github.com/popcornell/keras-triplet-center-loss/blob/master/runs/triplet_center_loss/Samples%20from%20Train%20Data%2C%20triplet_center_loss.png)