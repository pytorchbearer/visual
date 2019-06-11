Models
====================================

All the following models have been reimplemented to store layer outputs in :data:`visual.loss.LAYER_DICT`.
\* Pre-trained weights not available through torchvision.

.. list-table::
   :widths: 10 25
   :header-rows: 1

   * - Model
     - Specific

   * - AlexNet
     - alexnet

   * - DenseNet
     - densenet121, densenet161, densenet169, densenet201

   * - GoogLeNet
     - googlenet

   * - Inception
     - inception_v3

   * - MobileNet
     - mobilenet_v2

   * - ResNet
     - resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x8d, resnext50_32x4d

   * - ShuffleNet
     - shufflenet_v2_x0_5, shufflenet_v2_x1_0, \*shufflenet_v2_x1_5, \*shufflenet_v2_x2_0

   * - SqueezeNet
     - squeezenet1_0, squeezenet1_1

   * - VGG
     - vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn


..  automodule:: visual.models
	:members:
	:undoc-members: