# cifar10-ResNet

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
    training_epochs = 165
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4,
    batch_norm_momentum = 0.99,
    batch_norm_epsilon = 1e-3,
    batch_norm_center = True,
    batch_norm_scale = True



### Files
Source Files:

- resnet.py : main function that Initial images and model then, train model
- utils.py : use_gpu, normalization, load_images, build_optimizer    
- model.py : ResNet model, ResNetUnit
- outputs : graph, trained_model (h5 file)

## Accuracy
|Model|Validation Accuracy
|:------:|:---:|
|[VGG-16](https://github.com/SeHwanJoo/cifar10-vgg16)|93.15%|
|[ResNet-20](https://github.com/SeHwanJoo/cifar10-ResNet)|91.52%|
|[ResNet-32](https://github.com/SeHwanJoo/cifar10-ResNet)|92.53%|
|[ResNet-44](https://github.com/SeHwanJoo/cifar10-ResNet)|93.16%|
|[ResNet-56](https://github.com/SeHwanJoo/cifar10-ResNet)|93.21%|
|[ResNet-110](https://github.com/SeHwanJoo/cifar10-ResNet)|93.90%|

## Graph
### ResNet20
![resnet32_accuracy](https://user-images.githubusercontent.com/24911666/90999282-33678700-e601-11ea-9649-d8db5f198548.png)
![resnet32_loss](https://user-images.githubusercontent.com/24911666/90999285-3498b400-e601-11ea-9d19-1c2942264421.png)

### ResNet32
![resnet32_accuracy](https://user-images.githubusercontent.com/24911666/90999282-33678700-e601-11ea-9649-d8db5f198548.png)
![resnet32_loss](https://user-images.githubusercontent.com/24911666/90999285-3498b400-e601-11ea-9d19-1c2942264421.png)

### ResNet44
![resnet44_accuracy](https://user-images.githubusercontent.com/24911666/90999286-3498b400-e601-11ea-8b8d-86c915ae8f69.png)
![resnet44_loss](https://user-images.githubusercontent.com/24911666/90999287-35314a80-e601-11ea-8227-21318dc70bc1.png)

### ResNet56
![resnet56_accuracy](https://user-images.githubusercontent.com/24911666/90999289-35314a80-e601-11ea-8cee-5debd64a6cd5.png)
![resnet56_loss](https://user-images.githubusercontent.com/24911666/90999291-35c9e100-e601-11ea-8c3d-da1a5796c2c3.png)

###ResNet110
![resnet110_accuracypng](https://user-images.githubusercontent.com/24911666/90999293-35c9e100-e601-11ea-8734-bbbb26a7770b.png)
![resnet110_loss](https://user-images.githubusercontent.com/24911666/90999294-36627780-e601-11ea-9e04-048b285f94b7.png)