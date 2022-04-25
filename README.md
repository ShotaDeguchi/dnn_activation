# dnn_activation
This repo is a supplementary material for [author's blog post (Japanese)](link). We investigate the relationship between [DNN (Deep Neural Network) approximation](https://doi.org/10.1016/0893-6080(89)90020-8) and activation function selection. 

## Purpose
Activation functions introduce non-linearity to DNN approximation i.e. DNN approximations are heavily dependent on the properties of the selected activation functions. This repo's codes learns several functions with 5 different activation functions, namely, ReLU, ELU, Swish, sin, and tanh. Networks have different parameter initializations, [Glorot normal](https://proceedings.mlr.press/v9/glorot10a.html) for sin and tanh activation, [He normal](https://arxiv.org/abs/1502.01852) for ReLU, ELU, Swish. 

## Example
snapshots from examples. 

## Reference
[1] [author's blog post](link). 
<br>
[2] Kurt Hornik, Maxwell Stinchcombe, Halbert White: Multilayer feedforward networks are universal approximators, *Neural Networks*, Vol. 2, No. 5, pp. Pages 359-366, 1989. ([paper](https://doi.org/10.1016/0893-6080(89)90020-8))
<br>
[3] Glorot, X., Bengio, Y.: Understanding the difficulty of training deep feedforward neural networks, *Proceedings of Machine Learning Research*, Vol. 9, pp. 249-256, 2010. ([paper](https://proceedings.mlr.press/v9/glorot10a.html))
<br>
[4] He, K., Zhang, X., Ren, S., Sun, J.: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, *International Conference on Computer Vision (ICCV)*, pp. 1026-1034, 2015. ([paper](https://arxiv.org/abs/1502.01852))


