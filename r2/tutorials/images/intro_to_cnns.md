

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

下面6行代码使用常见模式定义卷积基数： [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) 和[MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)层的堆栈。

作为输入，CNN采用形状的张量（image_height, image_width, color_channels），忽略批量大小。MNIST有一个颜色通道（因为图像是灰度的），而彩色图像有三个颜色通道（R,G,B）。在此示例中，我们将配置CNN以处理形状（28,28,1）的输入，这是MNIST图像的格式，我们通过将参数input_shape传递给第一层来完成此操作。

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary() # 显示模型的架构
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
=================================================================
...
```

在上面，你可以看到每个Conv2D和MaxPooling2D层的输出都是3D张量的形状（高度，宽度，通道），随着我们在网络中更深入，宽度和高度大小趋于缩小，每个Conv2D层的输出通道的数由第一个参数（例如，32或64）控制。通常，随着宽度和高度的缩小，我们可以（计算地）在每个Conv2D层中添加更多输出通道

### 在顶部添加密集层

为了完成我们的模型，我们将最后的输出张量从卷积基（形状(3,3,64)）馈送到一个或多个密集层中以执行分类。密集层将矢量作为输入（1D），而当前输出是3D张量。首先，我们将3D输出展平（或展开）为1D，然后在顶部添加一个或多个Dense层。MINST有10个输出类，因此我们使用具有10输出和softmax激活的最终Dense层。

```
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary() # 显示模型的架构
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 576)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                36928     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
...
```

从上面可以看出，在通过两个密集层之前，我们的(3,3,64)输出被展平为矢量（576）。

### 编译和训练模型


```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
```

```
...
Epoch 5/5
60000/60000 [==============================] - 15s 258us/sample - loss: 0.0190 - accuracy: 0.9941
```

### 评估模型


```
test_loss, test_acc = model.evaluate(test_images, test_labels)
```
```
10000/10000 [==============================] - 1s 92us/sample - loss: 0.0272 - accuracy: 0.9921
```

```
print(test_acc)
```
```
0.9921
```
如你所见，我们简单的CNN已经达到了超过99%的测试精度，这几行代码还不错。另一种编写CNN的方式[here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/advanced.ipynb)（使用Keras Subclassing API和GradientTape）。
