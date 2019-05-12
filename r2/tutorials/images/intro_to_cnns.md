

# Convolutional Neural Networks

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/images/intro_to_cnns">
    <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
    View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/images/intro_to_cnns.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/images/intro_to_cnns.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

本教程演示了如何训练简单的[卷积神经网络](https://developers.google.com/machine-learning/glossary/#convolutional_neural_network)（CNN）来对MNIST数字进行分类。这个简单的网络将在MNIST测试集上实现99％以上的准确率。因为本教程使用[Keras Sequential API](https://www.tensorflow.org/guide/keras)，所以创建和训练我们的模型只需几行代码。

注意：CNN使用GPU训练更快。

### 导入TensorFlow


```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
```

### 下载预处理MNIST数据集


```
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 特征缩放[0, 1]区间 
train_images, test_images = train_images / 255.0, test_images / 255.0
```

### 创建卷积基

The 6 lines of code below define the convolutional base using a common pattern: a stack of [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. If you are new to color channels, MNIST has one (because the images are grayscale), whereas a color image has three (R,G,B). In this example, we will configure our CNN to process inputs of shape (28, 28, 1), which is the format of MNIST images. We do this by passing the argument `input_shape` to our first layer.

下面6行代码使用常见模式定义卷积基数： [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) 和[MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)层的堆栈。

作为输入，CNN采用形状的张量（image_height, image_width, color_channels），忽略批量大小，如果您不熟悉颜色通道，MNIST有一个（因为图像是灰度的），而彩色图像有三个（R,G,B）。在此示例中，我们将配置CNN以处理形状（28,28,1）的输入，这是MNIST图像的格式，我们通过将参数input_shape传递给第一层来完成此操作。




```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

Let display the architecture of our model so far.


```
model.summary()
```

Above, you can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). The width and height dimensions tend to shrink as we go deeper in the network. The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). Typically,  as the width and height shrink, we can afford (computationally) to add more output channels in each Conv2D layer.

### Add Dense layers on top
To complete our model, we will feed the last output tensor from the convolutional base (of shape (3, 3, 64)) into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. First, we will flatten (or unroll) the 3D output to 1D,  then add one or more Dense layers on top. MNIST has 10 output classes, so we use a final Dense layer with 10 outputs and a softmax activation.


```
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

 Here's the complete architecture of our model.


```
model.summary()
```

As you can see, our (3, 3, 64) outputs were flattened into vectors of shape (576) before going through two Dense layers.

### Compile and train the model


```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
```

### Evaluate the model


```
test_loss, test_acc = model.evaluate(test_images, test_labels)
```


```
print(test_acc)
```

As you can see, our simple CNN has achieved a test accuracy of over 99%. Not bad for a few lines of code! For another style of writing a CNN (using the Keras Subclassing API and a GradientTape) head [here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/advanced.ipynb).
