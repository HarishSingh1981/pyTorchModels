# Custom Resnet Network

This repository contains cutom Resnet architecture having 

## Architecture summary

```
Custom ResNet architecture for CIFAR10 that has the following architecture:
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
```

## Target for custom Architecture and usage of one cycle policy

```
The target of custom Resnet architecture is to achieve 93.8% accuracy in 24 epochs. We need to use OneCyclePolicy scheduler to train the network. To find the
optimal LR I have used "torch_lr_finder" module. The finding of optimal LR is achieved
1. First use the range test between 1e-07 to 1 with exp step and analyzed the loss curve. The module has suggested the best LR as 0.0152 but it shows multiple
slopes.
2. I have ran again range test between 1e-03 to 1 to again fine tune the best LR and took the new best LR suggested by the module [0.0614]
3. As per the target, in oneCyclePolicy we have achieved the optimal LR in 5th epoch hence the pct_start is calculated as 5/epochs [i.e. 0.2083]

```
