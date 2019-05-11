
##### Copyright 2019 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```


```
#@title MIT License
#
# Copyright (c) 2017 François Chollet                                                                                                                    # IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

# Transfer Learning Using Pretrained ConvNets

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/images/transfer_learning"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

In this tutorial you will learn how to classify cats vs dogs images by using transfer learning from a pre-trained network.


A **pre-trained model** is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as it is, or use **transfer learning** to customize this model to a given task.

The intuition behind transfer learning is that if a model trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch training a large model on a large dataset.

In this notebook, you will try two ways to customize a pretrained model:

1. **Feature Extraction**: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for our dataset.

  You do not need to (re)train the entire model.  The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to original classification task, and subsequently specific to the set of classes on which the model was trained.

2. **Fine-Tuning**: Unfreezing a few of the top layers of a frozen model base and jointly training both the newly-added classifier layers and the last layers of the base model. This allows us to "fine tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

You will follow the general machine learning workflow.

1. Examine and understand the data
2. Build an input pipeline, in this case using Keras `ImageDataGenerator`
3. Compose our model
  * Load in our pretrained base model (and pretrained weights)
  * Stack our classification layers on top
4. Train our model
5. Evaluate model



```
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

import matplotlib.pyplot as plt
```


```
!pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

keras = tf.keras
```

## Data preprocessing

### Data download

Use [TensorFlow Datasets](http://tensorflow.org/datasets) to load the cats and dogs dataset.

This `tfds` package is the easiest way to load pre-defined data. If you have your own data, and are interested in importing using it with TensroFlow see [loading image data](../load_data/images.ipynb)



```
import tensorflow_datasets as tfds
```

The `tfds.load` method downloads and caches the data, and returns a `tf.data.Dataset` obejct. These obejcts provide powerful, efficient methods for manipulating data and piping it into your model.

Since `"cats_vs_dog"` doesn't define standard splits, use the subsplit feature to divide it into (train, validation, test) with 80%, 10%, 10% of the data respectively.


```
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)
```

The resulting `tf.data.Dataset` objects contain `(image, label)` pairs. Where the images have variable shape and 3 channels, and the label is a scalar.


```
print(raw_train)
print(raw_validation)
print(raw_test)
```

    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>


Show the first two images and labels from the training set:


```
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
```


![png](transfer_learning_files/transfer_learning_17_0.png)



![png](transfer_learning_files/transfer_learning_17_1.png)


### Format the Data

Use the `tf.image` module to format the images for the task.

Resize the images to a fixes input size, and rescale the input channels to a range of `[-1,1]`

<!-- TODO(markdaoust): fix the keras_applications preprocessing functions to work in tf2 -->


```
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label
```

Apply this function to each item in the dataset using the map method:


```
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
```

Now shuffle and batch the data.


```
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
```


```
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
```

Inspect a batch of data:


```
for image_batch, label_batch in train_batches.take(1):
  pass

image_batch.shape
```




    TensorShape([32, 160, 160, 3])



## Create the base model from the pre-trained convnets
You will create the base model from the **MobileNet V2** model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images. ImageNet has a fairly arbitrary research training dataset with categories like `jackfruit` and `syringe`, but this base of knowledge will help us tell apart cats and dogs from our specific dataset.

First, you need to pick which layer of MobileNet V2 you will use for feature extraction. Obviously, the very last classification layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful.  Instead, you will follow the common practice to instead depend on the very last layer before the flatten operation. This layer is called the "bottleneck layer". The bottleneck features retain much generality as compared to the final/top layer.

First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the **include_top=False** argument, you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.


```
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

    Downloading data from https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
    9412608/9406464 [==============================] - 1s 0us/step


This feature extractor converts each `160x160x3` image to a `5x5x1280` block of features. See what it does to the example batch of images:


```
feature_batch = base_model(image_batch)
print(feature_batch.shape)
```

    (32, 5, 5, 1280)


## Feature extraction
You will freeze the convolutional base created from the previous step and use that as a feature extractor, add a classifier on top of it and train the top-level classifier.

### Freeze the convolutional base
It's important to freeze the convolutional based before you compile and train the model. By freezing (or setting `layer.trainable = False`), you prevent the weights in a given layer from being updated during training. MobileNet V2 has many layers, so setting the entire model's trainable flag to `False` will freeze all the layers.


```
base_model.trainable = False
```


```
# Let's take a look at the base model architecture
base_model.summary()
```

    Model: "mobilenetv2_1.00_160"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            [(None, 160, 160, 3) 0
    __________________________________________________________________________________________________
    Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_1[0][0]
    __________________________________________________________________________________________________
    Conv1 (Conv2D)                  (None, 80, 80, 32)   864         Conv1_pad[0][0]
    __________________________________________________________________________________________________
    bn_Conv1 (BatchNormalizationV1) (None, 80, 80, 32)   128         Conv1[0][0]
    __________________________________________________________________________________________________
    Conv1_relu (ReLU)               (None, 80, 80, 32)   0           bn_Conv1[0][0]
    __________________________________________________________________________________________________
    expanded_conv_depthwise (Depthw (None, 80, 80, 32)   288         Conv1_relu[0][0]
    __________________________________________________________________________________________________
    expanded_conv_depthwise_BN (Bat (None, 80, 80, 32)   128         expanded_conv_depthwise[0][0]
    __________________________________________________________________________________________________
    expanded_conv_depthwise_relu (R (None, 80, 80, 32)   0           expanded_conv_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    expanded_conv_project (Conv2D)  (None, 80, 80, 16)   512         expanded_conv_depthwise_relu[0][0
    __________________________________________________________________________________________________
    expanded_conv_project_BN (Batch (None, 80, 80, 16)   64          expanded_conv_project[0][0]
    __________________________________________________________________________________________________
    block_1_expand (Conv2D)         (None, 80, 80, 96)   1536        expanded_conv_project_BN[0][0]
    __________________________________________________________________________________________________
    block_1_expand_BN (BatchNormali (None, 80, 80, 96)   384         block_1_expand[0][0]
    __________________________________________________________________________________________________
    block_1_expand_relu (ReLU)      (None, 80, 80, 96)   0           block_1_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_1_pad (ZeroPadding2D)     (None, 81, 81, 96)   0           block_1_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_1_depthwise (DepthwiseCon (None, 40, 40, 96)   864         block_1_pad[0][0]
    __________________________________________________________________________________________________
    block_1_depthwise_BN (BatchNorm (None, 40, 40, 96)   384         block_1_depthwise[0][0]
    __________________________________________________________________________________________________
    block_1_depthwise_relu (ReLU)   (None, 40, 40, 96)   0           block_1_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_1_project (Conv2D)        (None, 40, 40, 24)   2304        block_1_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_1_project_BN (BatchNormal (None, 40, 40, 24)   96          block_1_project[0][0]
    __________________________________________________________________________________________________
    block_2_expand (Conv2D)         (None, 40, 40, 144)  3456        block_1_project_BN[0][0]
    __________________________________________________________________________________________________
    block_2_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_2_expand[0][0]
    __________________________________________________________________________________________________
    block_2_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_2_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_2_depthwise (DepthwiseCon (None, 40, 40, 144)  1296        block_2_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_2_depthwise_BN (BatchNorm (None, 40, 40, 144)  576         block_2_depthwise[0][0]
    __________________________________________________________________________________________________
    block_2_depthwise_relu (ReLU)   (None, 40, 40, 144)  0           block_2_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_2_project (Conv2D)        (None, 40, 40, 24)   3456        block_2_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_2_project_BN (BatchNormal (None, 40, 40, 24)   96          block_2_project[0][0]
    __________________________________________________________________________________________________
    block_2_add (Add)               (None, 40, 40, 24)   0           block_1_project_BN[0][0]
                                                                     block_2_project_BN[0][0]
    __________________________________________________________________________________________________
    block_3_expand (Conv2D)         (None, 40, 40, 144)  3456        block_2_add[0][0]
    __________________________________________________________________________________________________
    block_3_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_3_expand[0][0]
    __________________________________________________________________________________________________
    block_3_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_3_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_3_pad (ZeroPadding2D)     (None, 41, 41, 144)  0           block_3_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_3_depthwise (DepthwiseCon (None, 20, 20, 144)  1296        block_3_pad[0][0]
    __________________________________________________________________________________________________
    block_3_depthwise_BN (BatchNorm (None, 20, 20, 144)  576         block_3_depthwise[0][0]
    __________________________________________________________________________________________________
    block_3_depthwise_relu (ReLU)   (None, 20, 20, 144)  0           block_3_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_3_project (Conv2D)        (None, 20, 20, 32)   4608        block_3_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_3_project_BN (BatchNormal (None, 20, 20, 32)   128         block_3_project[0][0]
    __________________________________________________________________________________________________
    block_4_expand (Conv2D)         (None, 20, 20, 192)  6144        block_3_project_BN[0][0]
    __________________________________________________________________________________________________
    block_4_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_4_expand[0][0]
    __________________________________________________________________________________________________
    block_4_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_4_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_4_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_4_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_4_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_4_depthwise[0][0]
    __________________________________________________________________________________________________
    block_4_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_4_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_4_project (Conv2D)        (None, 20, 20, 32)   6144        block_4_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_4_project_BN (BatchNormal (None, 20, 20, 32)   128         block_4_project[0][0]
    __________________________________________________________________________________________________
    block_4_add (Add)               (None, 20, 20, 32)   0           block_3_project_BN[0][0]
                                                                     block_4_project_BN[0][0]
    __________________________________________________________________________________________________
    block_5_expand (Conv2D)         (None, 20, 20, 192)  6144        block_4_add[0][0]
    __________________________________________________________________________________________________
    block_5_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_5_expand[0][0]
    __________________________________________________________________________________________________
    block_5_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_5_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_5_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_5_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_5_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_5_depthwise[0][0]
    __________________________________________________________________________________________________
    block_5_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_5_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_5_project (Conv2D)        (None, 20, 20, 32)   6144        block_5_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_5_project_BN (BatchNormal (None, 20, 20, 32)   128         block_5_project[0][0]
    __________________________________________________________________________________________________
    block_5_add (Add)               (None, 20, 20, 32)   0           block_4_add[0][0]
                                                                     block_5_project_BN[0][0]
    __________________________________________________________________________________________________
    block_6_expand (Conv2D)         (None, 20, 20, 192)  6144        block_5_add[0][0]
    __________________________________________________________________________________________________
    block_6_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_6_expand[0][0]
    __________________________________________________________________________________________________
    block_6_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_6_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_6_pad (ZeroPadding2D)     (None, 21, 21, 192)  0           block_6_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_6_depthwise (DepthwiseCon (None, 10, 10, 192)  1728        block_6_pad[0][0]
    __________________________________________________________________________________________________
    block_6_depthwise_BN (BatchNorm (None, 10, 10, 192)  768         block_6_depthwise[0][0]
    __________________________________________________________________________________________________
    block_6_depthwise_relu (ReLU)   (None, 10, 10, 192)  0           block_6_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_6_project (Conv2D)        (None, 10, 10, 64)   12288       block_6_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_6_project_BN (BatchNormal (None, 10, 10, 64)   256         block_6_project[0][0]
    __________________________________________________________________________________________________
    block_7_expand (Conv2D)         (None, 10, 10, 384)  24576       block_6_project_BN[0][0]
    __________________________________________________________________________________________________
    block_7_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_7_expand[0][0]
    __________________________________________________________________________________________________
    block_7_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_7_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_7_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_7_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_7_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_7_depthwise[0][0]
    __________________________________________________________________________________________________
    block_7_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_7_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_7_project (Conv2D)        (None, 10, 10, 64)   24576       block_7_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_7_project_BN (BatchNormal (None, 10, 10, 64)   256         block_7_project[0][0]
    __________________________________________________________________________________________________
    block_7_add (Add)               (None, 10, 10, 64)   0           block_6_project_BN[0][0]
                                                                     block_7_project_BN[0][0]
    __________________________________________________________________________________________________
    block_8_expand (Conv2D)         (None, 10, 10, 384)  24576       block_7_add[0][0]
    __________________________________________________________________________________________________
    block_8_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_8_expand[0][0]
    __________________________________________________________________________________________________
    block_8_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_8_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_8_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_8_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_8_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_8_depthwise[0][0]
    __________________________________________________________________________________________________
    block_8_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_8_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_8_project (Conv2D)        (None, 10, 10, 64)   24576       block_8_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_8_project_BN (BatchNormal (None, 10, 10, 64)   256         block_8_project[0][0]
    __________________________________________________________________________________________________
    block_8_add (Add)               (None, 10, 10, 64)   0           block_7_add[0][0]
                                                                     block_8_project_BN[0][0]
    __________________________________________________________________________________________________
    block_9_expand (Conv2D)         (None, 10, 10, 384)  24576       block_8_add[0][0]
    __________________________________________________________________________________________________
    block_9_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_9_expand[0][0]
    __________________________________________________________________________________________________
    block_9_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_9_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_9_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_9_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_9_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_9_depthwise[0][0]
    __________________________________________________________________________________________________
    block_9_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_9_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_9_project (Conv2D)        (None, 10, 10, 64)   24576       block_9_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_9_project_BN (BatchNormal (None, 10, 10, 64)   256         block_9_project[0][0]
    __________________________________________________________________________________________________
    block_9_add (Add)               (None, 10, 10, 64)   0           block_8_add[0][0]
                                                                     block_9_project_BN[0][0]
    __________________________________________________________________________________________________
    block_10_expand (Conv2D)        (None, 10, 10, 384)  24576       block_9_add[0][0]
    __________________________________________________________________________________________________
    block_10_expand_BN (BatchNormal (None, 10, 10, 384)  1536        block_10_expand[0][0]
    __________________________________________________________________________________________________
    block_10_expand_relu (ReLU)     (None, 10, 10, 384)  0           block_10_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_10_depthwise (DepthwiseCo (None, 10, 10, 384)  3456        block_10_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_10_depthwise_BN (BatchNor (None, 10, 10, 384)  1536        block_10_depthwise[0][0]
    __________________________________________________________________________________________________
    block_10_depthwise_relu (ReLU)  (None, 10, 10, 384)  0           block_10_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_10_project (Conv2D)       (None, 10, 10, 96)   36864       block_10_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_10_project_BN (BatchNorma (None, 10, 10, 96)   384         block_10_project[0][0]
    __________________________________________________________________________________________________
    block_11_expand (Conv2D)        (None, 10, 10, 576)  55296       block_10_project_BN[0][0]
    __________________________________________________________________________________________________
    block_11_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_11_expand[0][0]
    __________________________________________________________________________________________________
    block_11_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_11_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_11_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_11_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_11_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_11_depthwise[0][0]
    __________________________________________________________________________________________________
    block_11_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_11_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_11_project (Conv2D)       (None, 10, 10, 96)   55296       block_11_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_11_project_BN (BatchNorma (None, 10, 10, 96)   384         block_11_project[0][0]
    __________________________________________________________________________________________________
    block_11_add (Add)              (None, 10, 10, 96)   0           block_10_project_BN[0][0]
                                                                     block_11_project_BN[0][0]
    __________________________________________________________________________________________________
    block_12_expand (Conv2D)        (None, 10, 10, 576)  55296       block_11_add[0][0]
    __________________________________________________________________________________________________
    block_12_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_12_expand[0][0]
    __________________________________________________________________________________________________
    block_12_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_12_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_12_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_12_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_12_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_12_depthwise[0][0]
    __________________________________________________________________________________________________
    block_12_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_12_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_12_project (Conv2D)       (None, 10, 10, 96)   55296       block_12_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_12_project_BN (BatchNorma (None, 10, 10, 96)   384         block_12_project[0][0]
    __________________________________________________________________________________________________
    block_12_add (Add)              (None, 10, 10, 96)   0           block_11_add[0][0]
                                                                     block_12_project_BN[0][0]
    __________________________________________________________________________________________________
    block_13_expand (Conv2D)        (None, 10, 10, 576)  55296       block_12_add[0][0]
    __________________________________________________________________________________________________
    block_13_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_13_expand[0][0]
    __________________________________________________________________________________________________
    block_13_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_13_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_13_pad (ZeroPadding2D)    (None, 11, 11, 576)  0           block_13_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_13_depthwise (DepthwiseCo (None, 5, 5, 576)    5184        block_13_pad[0][0]
    __________________________________________________________________________________________________
    block_13_depthwise_BN (BatchNor (None, 5, 5, 576)    2304        block_13_depthwise[0][0]
    __________________________________________________________________________________________________
    block_13_depthwise_relu (ReLU)  (None, 5, 5, 576)    0           block_13_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_13_project (Conv2D)       (None, 5, 5, 160)    92160       block_13_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_13_project_BN (BatchNorma (None, 5, 5, 160)    640         block_13_project[0][0]
    __________________________________________________________________________________________________
    block_14_expand (Conv2D)        (None, 5, 5, 960)    153600      block_13_project_BN[0][0]
    __________________________________________________________________________________________________
    block_14_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_14_expand[0][0]
    __________________________________________________________________________________________________
    block_14_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_14_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_14_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_14_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_14_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_14_depthwise[0][0]
    __________________________________________________________________________________________________
    block_14_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_14_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_14_project (Conv2D)       (None, 5, 5, 160)    153600      block_14_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_14_project_BN (BatchNorma (None, 5, 5, 160)    640         block_14_project[0][0]
    __________________________________________________________________________________________________
    block_14_add (Add)              (None, 5, 5, 160)    0           block_13_project_BN[0][0]
                                                                     block_14_project_BN[0][0]
    __________________________________________________________________________________________________
    block_15_expand (Conv2D)        (None, 5, 5, 960)    153600      block_14_add[0][0]
    __________________________________________________________________________________________________
    block_15_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_15_expand[0][0]
    __________________________________________________________________________________________________
    block_15_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_15_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_15_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_15_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_15_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_15_depthwise[0][0]
    __________________________________________________________________________________________________
    block_15_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_15_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_15_project (Conv2D)       (None, 5, 5, 160)    153600      block_15_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_15_project_BN (BatchNorma (None, 5, 5, 160)    640         block_15_project[0][0]
    __________________________________________________________________________________________________
    block_15_add (Add)              (None, 5, 5, 160)    0           block_14_add[0][0]
                                                                     block_15_project_BN[0][0]
    __________________________________________________________________________________________________
    block_16_expand (Conv2D)        (None, 5, 5, 960)    153600      block_15_add[0][0]
    __________________________________________________________________________________________________
    block_16_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_16_expand[0][0]
    __________________________________________________________________________________________________
    block_16_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_16_expand_BN[0][0]
    __________________________________________________________________________________________________
    block_16_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_16_expand_relu[0][0]
    __________________________________________________________________________________________________
    block_16_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_16_depthwise[0][0]
    __________________________________________________________________________________________________
    block_16_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_16_depthwise_BN[0][0]
    __________________________________________________________________________________________________
    block_16_project (Conv2D)       (None, 5, 5, 320)    307200      block_16_depthwise_relu[0][0]
    __________________________________________________________________________________________________
    block_16_project_BN (BatchNorma (None, 5, 5, 320)    1280        block_16_project[0][0]
    __________________________________________________________________________________________________
    Conv_1 (Conv2D)                 (None, 5, 5, 1280)   409600      block_16_project_BN[0][0]
    __________________________________________________________________________________________________
    Conv_1_bn (BatchNormalizationV1 (None, 5, 5, 1280)   5120        Conv_1[0][0]
    __________________________________________________________________________________________________
    out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]
    ==================================================================================================
    Total params: 2,257,984
    Trainable params: 0
    Non-trainable params: 2,257,984
    __________________________________________________________________________________________________


### Add a classification head

To generate predictions from the block of features, average over the spatial `5x5` spatial locations, using a `tf.keras.layers.GlobalAveragePooling2D` layer to convert the features to  a single 1280-element vector per image.


```
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
```

    (32, 1280)


Apply a `tf.keras.layers.Dense` layer to convert these features into a single prediction per image. You don't need an activation function here because this prediction will be treated as a `logit`, or a raw prediciton value.  Positive numbers predict class 1, negative numbers predict class 0.


```
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
```

    (32, 1)


Now stack the feature extractor, and these two layers using a `tf.keras.Sequential` model:


```
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```

### Compile the model

You must compile the model before training it.  Since there are two classes, use a binary cross-entropy loss.


```
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```


```
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    mobilenetv2_1.00_160 (Model) (None, 5, 5, 1280)        2257984
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 1280)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 1281
    =================================================================
    Total params: 2,259,265
    Trainable params: 1,281
    Non-trainable params: 2,257,984
    _________________________________________________________________


The 2.5M parameters in MobileNet are frozen, but there are 1.2K _trainable_ parameters in the Dense layer.  These are divided between two `tf.Variable` objects, the weights and biases.


```
len(model.trainable_variables)
```




    2



### Train the model

After training for 10 epochs, you should see ~96% accuracy.

<!-- TODO(markdaoust): delete steps_per_epoch in TensorFlow r1.14/r2.0 -->


```
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)
```


```
initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
```

    20/20 [==============================] - 4s 219ms/step - loss: 3.1885 - accuracy: 0.6109



```
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
```

    initial loss: 3.19
    initial accuracy: 0.61



```
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
```

    Epoch 1/10
    581/581 [==============================] - 102s 175ms/step - loss: 1.8917 - accuracy: 0.7606 - val_loss: 0.8860 - val_accuracy: 0.8828
    Epoch 2/10
    581/581 [==============================] - 94s 161ms/step - loss: 0.9989 - accuracy: 0.8697 - val_loss: 0.4330 - val_accuracy: 0.9281
    Epoch 3/10
    581/581 [==============================] - 99s 170ms/step - loss: 0.7417 - accuracy: 0.8995 - val_loss: 0.2629 - val_accuracy: 0.9469
    Epoch 4/10
    581/581 [==============================] - 94s 162ms/step - loss: 0.6681 - accuracy: 0.9133 - val_loss: 0.2753 - val_accuracy: 0.9563
    Epoch 5/10
    581/581 [==============================] - 98s 169ms/step - loss: 0.5944 - accuracy: 0.9220 - val_loss: 0.2410 - val_accuracy: 0.9641
    Epoch 6/10
    581/581 [==============================] - 98s 169ms/step - loss: 0.5779 - accuracy: 0.9252 - val_loss: 0.2003 - val_accuracy: 0.9656
    Epoch 7/10
    581/581 [==============================] - 94s 162ms/step - loss: 0.5151 - accuracy: 0.9328 - val_loss: 0.2115 - val_accuracy: 0.9688
    Epoch 8/10
    581/581 [==============================] - 95s 164ms/step - loss: 0.5076 - accuracy: 0.9346 - val_loss: 0.1678 - val_accuracy: 0.9703
    Epoch 9/10
    581/581 [==============================] - 96s 165ms/step - loss: 0.4679 - accuracy: 0.9368 - val_loss: 0.1668 - val_accuracy: 0.9703
    Epoch 10/10
    581/581 [==============================] - 96s 165ms/step - loss: 0.4921 - accuracy: 0.9381 - val_loss: 0.1847 - val_accuracy: 0.9719


### Learning curves

Let's take a look at the learning curves of the training and validation accuracy/loss when using the MobileNet V2 base model as a fixed feature extractor.


```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](transfer_learning_files/transfer_learning_53_0.png)


Note: If you are wondering why the validation metrics are clearly better than the training metrics, the main factor is because layers like `tf.keras.layers.BatchNormalization` and `tf.keras.layers.Dropout` affect accuracy during training. They are turned off when calculating validation loss.

To a lesser extent, it is also because training metrics report the average for an epoch, while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.

## Fine tuning
In our feature extraction experiment, you were only training a few layers on top of an MobileNet V2 base model. The weights of the pre-trained network were **not** updated during training.

One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset.

Note: This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.

Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features which generalize to almost all types of images. As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

### Un-freeze the top layers of the model


All you need to do is unfreeze the `base_model` and set the bottom layers be un-trainable. Then, you should recompile the model (necessary for these changes to take effect), and resume training.


```
base_model.trainable = True
```


```
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
```

    Number of layers in the base model:  155


### Compile the model

Compile the model using a much lower training rate.


```
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
```


```
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    mobilenetv2_1.00_160 (Model) (None, 5, 5, 1280)        2257984
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 1280)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 1281
    =================================================================
    Total params: 2,259,265
    Trainable params: 1,863,873
    Non-trainable params: 395,392
    _________________________________________________________________



```
len(model.trainable_variables)
```




    58



### Continue Train the model

If you trained to convergence earlier, this will get you a few percent more accuracy.


```
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch = initial_epochs,
                         validation_data=validation_batches)
```

    Epoch 11/20
    581/581 [==============================] - 134s 231ms/step - loss: 0.4208 - accuracy: 0.9484 - val_loss: 0.1907 - val_accuracy: 0.9812
    Epoch 12/20
    581/581 [==============================] - 114s 197ms/step - loss: 0.3359 - accuracy: 0.9570 - val_loss: 0.1835 - val_accuracy: 0.9844
    Epoch 13/20
    581/581 [==============================] - 116s 200ms/step - loss: 0.2930 - accuracy: 0.9650 - val_loss: 0.1505 - val_accuracy: 0.9844
    Epoch 14/20
    581/581 [==============================] - 114s 196ms/step - loss: 0.2561 - accuracy: 0.9701 - val_loss: 0.1575 - val_accuracy: 0.9859
    Epoch 15/20
    581/581 [==============================] - 119s 206ms/step - loss: 0.2302 - accuracy: 0.9715 - val_loss: 0.1600 - val_accuracy: 0.9812
    Epoch 16/20
    581/581 [==============================] - 115s 197ms/step - loss: 0.2134 - accuracy: 0.9747 - val_loss: 0.1407 - val_accuracy: 0.9828
    Epoch 17/20
    581/581 [==============================] - 115s 197ms/step - loss: 0.1546 - accuracy: 0.9813 - val_loss: 0.0944 - val_accuracy: 0.9828
    Epoch 18/20
    581/581 [==============================] - 116s 200ms/step - loss: 0.1636 - accuracy: 0.9794 - val_loss: 0.0947 - val_accuracy: 0.9844
    Epoch 19/20
    581/581 [==============================] - 115s 198ms/step - loss: 0.1356 - accuracy: 0.9823 - val_loss: 0.1169 - val_accuracy: 0.9828
    Epoch 20/20
    581/581 [==============================] - 116s 199ms/step - loss: 0.1243 - accuracy: 0.9849 - val_loss: 0.1121 - val_accuracy: 0.9875


Let's take a look at the learning curves of the training and validation accuracy/loss, when fine tuning the last few layers of the MobileNet V2 base model and training the classifier on top of it. The validation loss is much higher than the training loss, so you may get some overfitting.

You may also get some overfitting as the new training set is relatively small and similar to the original MobileNet V2 datasets.


After fine tuning the model nearly reaches 98% accuracy.


```
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
```


```
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](transfer_learning_files/transfer_learning_70_0.png)


## Summary:

* **Using a pre-trained model for feature extraction**:  When working with a small dataset, it is common to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier get updated during training.
In this case, the convolutional base extracted all the features associated with each image and you just trained a classifier that determines the image class given that set of extracted features.

* **Fine-tuning a pre-trained model**: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning.
In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the orginial dataset that the pre-trained model was trained on.

