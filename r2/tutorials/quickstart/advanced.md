---
title: 专家入门TensorFlow 2.0
categories: tensorflow2官方教程
tags: tensorflow2.0教程
top: 1906
abbrlink: tensorflow/tf2-tutorials-quickstart-advanced
---

# 专家入门TensorFlow 2.0使用流程：数据处理、自定义模型、损失、指标、梯度下降 (tensorflow2.0官方教程翻译)

初学者入门教程中，使用tf.keras.Sequential模型，只是简单的堆叠模型。
本文是专家级入门，使用 Keras 模型子类 API 构建模型，会使用更底层一点的的函数接口，自定义模型、损失、评估指标和梯度下降控制等，流程清晰。


开始，请将TensorFlow库导入您的程序：

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # 安装命令 `pip install tensorflow-gpu==2.0.0-alpha0`

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
```

加载并准备[MNIST数据集](http://yann.lecun.com/exdb/mnist/).。

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 添加一个通道维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```

使用tf.data批处理和随机打乱数据集：

```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

通过使用Keras[模型子类 API](https://tensorflow.google.cn/guide/keras#model_subclassing)构建`tf.keras`模型：

```python
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()
```

选择优化器和损失函数进行训练：

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()
```

选择指标（metrics）以衡量模型的损失和准确性。这些指标累积超过周期的值，然后打印整体结果。

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

使用`tf.GradientTape`训练模型：

```python
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
```

现在测试模型：

```python
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
```

```python
EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
```

```
      Epoch 1, Loss: 0.13177014887332916, Accuracy: 96.06000518798828, Test Loss: 0.05814294517040253, Test Accuracy: 98.04999542236328 
      ...
      Epoch 5, Loss: 0.042211469262838364, Accuracy: 98.72000122070312, Test Loss: 0.05708516761660576, Test Accuracy: 98.3239974975586
```

现在，图像分类器在该数据集上的准确度达到约98％。要了解更多信息，请阅读 [TensorFlow教程](https://tensorflow.google.cn/beta/tutorials/keras).。

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-tutorials-quickstart-advanced.html](https://www.mashangxue123.com/tensorflow/tf2-tutorials-quickstart-advanced.html)
> 英文版本：[https://tensorflow.google.cn/beta/tutorials/quickstart/advanced](https://tensorflow.google.cn/beta/tutorials/quickstart/advanced)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/quickstart/beginner.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/quickstart/advanced.md)
