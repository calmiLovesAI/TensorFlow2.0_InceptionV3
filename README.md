# TensorFlow2.0_InceptionV3
A TensorFlow_2.0 implementation of InceptionV3.

## Train
1. Requirements:
+ Python 3.6
+ Tensorflow 2.0.0-beta1
2. To train the ResNet on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set. The dataset directory will be like this:
 ```
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
```
4. Change the corresponding parameters in **config.py**.
5. Run **train.py** to start training.
## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.


## References
1. The original paper :https://arxiv.org/abs/1512.00567
2. Google official implementation of InceptionV3 (TensorFlow 1.x): https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
3. https://www.jianshu.com/p/3bbf0675cfce
4. Official PyTorch implementation of InceptionV3 : https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
