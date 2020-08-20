# cifar10-ResNet

## Description


## Training
Trained using two approaches for 250 epochs:
1. Keeping the base model's layer fixed, and
2. By training end-to-end

## Model Summary
### ResNet 20
  
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 16)        432
    _________________________________________________________________
    sequential (Sequential)      (None, 32, 32, 16)        14208
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 16, 16, 32)        51392
    _________________________________________________________________
    sequential_2 (Sequential)    (None, 8, 8, 64)          204160
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 64)          256
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 1, 1, 10)          650
    _________________________________________________________________
    activation (Activation)      (None, 10)                0
    =================================================================
    Total params: 271,098
    Trainable params: 269,722
    Non-trainable params: 1,376
    _________________________________________________________________
### ResNet 32
    
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 16)        432
    _________________________________________________________________
    sequential (Sequential)      (None, 32, 32, 16)        23680
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 16, 16, 32)        88768
    _________________________________________________________________
    sequential_2 (Sequential)    (None, 8, 8, 64)          352640
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 64)          256
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 1, 1, 10)          650
    _________________________________________________________________
    activation (Activation)      (None, 10)                0
    =================================================================
    Total params: 466,426
    Trainable params: 464,154
    Non-trainable params: 2,272
    _________________________________________________________________
### ResNet 44

    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 16)        432
    _________________________________________________________________
    sequential (Sequential)      (None, 32, 32, 16)        33152
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 16, 16, 32)        126144
    _________________________________________________________________
    sequential_2 (Sequential)    (None, 8, 8, 64)          501120
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 64)          256
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 1, 1, 10)          650
    _________________________________________________________________
    activation (Activation)      (None, 10)                0
    =================================================================
    Total params: 661,754
    Trainable params: 658,586
    Non-trainable params: 3,168
    _________________________________________________________________

 ### ResNet 56
 
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 16)        432
    _________________________________________________________________
    sequential (Sequential)      (None, 32, 32, 16)        42624
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 16, 16, 32)        163520
    _________________________________________________________________
    sequential_2 (Sequential)    (None, 8, 8, 64)          649600
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 64)          256
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 1, 1, 10)          650
    _________________________________________________________________
    activation (Activation)      (None, 10)                0
    =================================================================
    Total params: 857,082
    Trainable params: 853,018
    Non-trainable params: 4,064
    _________________________________________________________________

### ResNet 110

    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 16)        432
    _________________________________________________________________
    sequential (Sequential)      (None, 32, 32, 16)        85248
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 16, 16, 32)        331712
    _________________________________________________________________
    sequential_2 (Sequential)    (None, 8, 8, 64)          1317760
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8, 8, 64)          256
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 1, 1, 10)          650
    _________________________________________________________________
    activation (Activation)      (None, 10)                0
    =================================================================
    Total params: 1,736,058
    Trainable params: 1,727,962
    Non-trainable params: 8,096
    _________________________________________________________________


## Hyper parameter
    training_epochs = 250
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_decay = 1e-6
    lr_drop = 20



#### Files
Source Files:

- vgg16.py
    - load_images() : load cifar-10 images (train, test)
    - normalization() : normalization cifar-10 images
    - ResNetUnit : create multiple shortcut layer with Conv2D, BatchNormalization,
    - ResNetModel  : create deep learning model based vgg16
    - train() : train VGG16Model with cifar-10 images
    - main() : main function that Initial images and model then, call train function
    
    
- cifar10-resnet.h5 : trained model's weights


|Model|Validation Accuracy
|:------:|:---:|
|[VGG-16](https://github.com/SeHwanJoo/cifar10-vgg16)|93.15%|
|[ResNet-20](https://github.com/SeHwanJoo/cifar10-ResNet)||
|[ResNet-32](https://github.com/SeHwanJoo/cifar10-ResNet)||
|[ResNet-44](https://github.com/SeHwanJoo/cifar10-ResNet)||
|[ResNet-56](https://github.com/SeHwanJoo/cifar10-ResNet)||
|[ResNet-110](https://github.com/SeHwanJoo/cifar10-ResNet)||
