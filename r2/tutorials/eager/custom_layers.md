---
title: 自定义层 (tensorflow2官方教程翻译）
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 1999
abbrlink: tensorflow/tf2-tutorials-eager-custom_layers
---

# 自定义层

> 最新版本：[http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-custom_layers](http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-custom_layers)
> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/eager/custom_layers](https://tensorflow.google.cn/alpha/tutorials/eager/custom_layers)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/custom_layers.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/custom_layers.md)


我们建议使用 `tf.keras` 作为构建神经网络的高级API，也就是说，大多数TensorFlow API都可用于Eager execution。
 
```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
```

## 1. 对图层的常用操作

在编写机器学习模型的代码时，大多数情况下，您希望以比单个操作和单个变量操作更高的抽象级别上进行操作。

许多机器学习模型都可以表示为相对简单的层的组合和叠加，TensorFlow提供了一组公共层和一种简单的方法，让您可以从头开始编写自己的特定于应用程序的层，也可以表示为现有层的组合。

TensorFlow在 `tf.keras` 中包含完整 [Keras](https://keras.io) API，而Keras层在构建自己的模型时非常有用。


```python
# 在tf.keras.layers包中，图层是对象。要构造一个图层，只需构造一个对象。 
# 大多数层将输出维度/通道的数量作为第一个参数。 
layer = tf.keras.layers.Dense(100)

# 输入维度的数量通常是不必要的，因为它可以在第一次使用层时推断出来， 
# 但如果您想手动指定它，则可以提供它，这在某些复杂模型中很有用。 
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
```

可以在文档([链接](https://www.tensorflow.org/api_docs/python/tf/keras/layers))中看到预先存在的层的完整列表，它包括Dense（完全连接层），Conv2D，LSTM，BatchNormalization，Dropout等等。

```python
# 要使用图层，只需调用它即可。 
layer(tf.zeros([10, 5]))
```


```python
# 层有许多有用的方法，例如，您可以使用 `layer.variables` 和可训练变量使用 
# `layer.trainable_variables`检查图层中的所有变量，在这种情况下， 
# 完全连接的层将具有权重和偏差的变量。 
print(layer.variables) 
```

```python
# 变量也可以通过nice accessors访问
print(layer.kernel, layer.bias)
```

## 2. 使用keras实现自定义层

实现自己的层的最佳方法是扩展`tf.keras.Layer` 类并实现：

  *  `__init__` ，您可以在其中执行所有与输入无关的初始化

  * `build`，您可以在其中了解输入张量的形状，并可以执行其余的初始化

  * `call`，在那里进行正向计算。


请注意，您不必等到调用 `build` 来创建变量，您也可以在 `__init__`中创建它们。但是，在 `build` 中创建它们的好处是，它支持根据将要操作的层的输入形状，创建后期变量。另一方面，在 `__init__` 中创建变量意味着需要明确指定创建变量所需的形状。

```python
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)
```

如果尽可能使用标准层，则整体代码更易于阅读和维护，因为其他读者将熟悉标准层的行为。如果你想使用 `tf.keras.layers` 中不存在的图层，请考虑提交[github问题](http://github.com/tensorflow/tensorflow/issues/new)，或者最好向我们发送pull request！


## 3. 通过组合层构建模型

在机器学习模型中，许多有趣的类似层的事物都是通过组合现有层来实现的。例如，resnet中的每个残差块都是convolutions、 batch normalizations和shortcut的组合。

创建包含其他层的类似层的事物时使用的主类是 `tf.keras.Model`，实现一个是通过继承自 `tf.keras.Model` 完成的。

```python
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])
```

```
      tf.Tensor( [[[[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]] [[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]]]], shape=(1, 2, 3, 3), dtype=float32)
      ['resnet_identity_block/conv2d/kernel:0', 'resnet_identity_block/conv2d/bias:0',
      'resnet_identity_block/batch_normalization_v2/gamma:0', 'resnet_identity_block/batch_normalization_v2/beta:0',
      'resnet_identity_block/conv2d_1/kernel:0', 'resnet_identity_block/conv2d_1/bias:0',
      'resnet_identity_block/batch_normalization_v2_1/gamma:0', 'resnet_identity_block/batch_normalization_v2_1/beta:0',
      'resnet_identity_block/conv2d_2/kernel:0', 'resnet_identity_block/conv2d_2/bias:0',
      'resnet_identity_block/batch_normalization_v2_2/gamma:0', 'resnet_identity_block/batch_normalization_v2_2/beta:0']
```

然而，在大多数情况下，组成许多层的模型只是简单地调用一个又一个层。这可以通过使用 `tf.keras.Sequential`在很少的代码中完成

```python
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1),
                                                    input_shape=(
                                                        None, None, 3)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(2, 1,
                                                    padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(3, (1, 1)),
                             tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))
```

# 4. 下一步

现在，您可以返回到之前的教程，并调整线性回归示例，以使用更好的结构化层和模型。

> 最新版本：[http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-custom_layers](http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-custom_layers)
> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/eager/custom_layers](https://tensorflow.google.cn/alpha/tutorials/eager/custom_layers)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/custom_layers.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/custom_layers.md)

