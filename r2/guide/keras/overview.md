---
title: Keras概述：构建模型，输入数据，训练，评估，回调
tags: tensorflow2.0教程
categories: tensorflow2官方教程
top: 1999
abbrlink: tensorflow/tf2-guide-keras-overview
---

# Keras概述：构建模型，输入数据，训练，评估，回调，保存，分布(tensorflow2.0官方教程翻译）

Keras 是一个用于构建和训练深度学习模型的高阶API。它可用于快速设计原型、高级研究和生产，具有以下三个主要优势：

* 方便用户使用

Keras 具有针对常见用例做出优化的简单而一致的界面。它可针对用户错误提供切实可行的清晰反馈。

* 模块化和可组合

将可配置的构造块连接在一起就可以构建 Keras 模型，并且几乎不受限制。

* 易于扩展

可以编写自定义构造块以表达新的研究创意，并且可以创建新层、损失函数并开发先进的模型。

## 1. 导入 tf.keras

`tf.keras` 是 TensorFlow 对 [Keras API 规范](https://keras.io)的实现。这是一个用于构建和训练模型的高阶 API，包含对 TensorFlow 特定功能（例如[eager execution](https://tensorflow.google.cn/guide/keras#eager_execution)、[`tf.data` 管道](https://tensorflow.google.cn/api_docs/python/tf/data)和 [Estimators](https://tensorflow.google.cn/guide/estimators)）的顶级支持。 `tf.keras` 使 TensorFlow 更易于使用，并且不会牺牲灵活性和性能。

首先，导入 `tf.keras` 以设置 TensorFlow 程序：

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
```

`tf.keras` 可以运行任何与 Keras 兼容的代码，但请注意：

* 最新版 TensorFlow 中的 `tf.keras` 版本可能与 PyPI 中的最新 keras 版本不同。请查看 `tf.keras.version`。

* [保存模型的权重](#weights_only)时，`tf.keras` 默认采用检查点格式。请传递 ` save_format='h5' `以使用 HDF5。

## 2. 构建简单的模型

### 2.1. 序列模型

在 Keras 中，您可以通过组合层来构建模型。模型（通常）是由层构成的图。最常见的模型类型是层的堆叠：`tf.keras.Sequential` 模型。

要构建一个简单的全连接网络（即多层感知器），请运行以下代码：

```python
from tensorflow.keras import layers

model = tf.keras.Sequential()
# 向模型添加一个64单元的密集连接层：
model.add(layers.Dense(64, activation='relu'))
# 加上另一个：
model.add(layers.Dense(64, activation='relu'))
# 添加一个包含10个输出单位的softmax层：
model.add(layers.Dense(10, activation='softmax'))
```

您可以找到有关如何使用Sequential模型的完整简短示例 [here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/beginner.ipynb).

要了解如何构建比Sequential模型更高级的模型，请参阅:
- [Guide to the Keras Functional](https://tensorflow.google.cn/alpha/guide/keras/functional)
- [Guide to writing layers and models from scratch with subclassing](https://tensorflow.google.cn/alpha/guide/keras/custom_layers_and_models)

### 2.2. 配置层

我们可以使用很多 `tf.keras.layers`，它们具有一些相同的构造函数参数：

* `activation`：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。

* `kernel_initializer` 和 `bias_initializer`：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。

* `kernel_regularizer` 和 `bias_regularizer`：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。

以下代码使用构造函数参数实例化 `tf.keras.layers. Dense` 层：

```python
# 创建一个sigmoid层:
layers.Dense(64, activation='sigmoid')
# 或者使用下面的代码创建:
layers.Dense(64, activation=tf.keras.activations.sigmoid)

# 将具有因子0.01的L1正则化的线性层应用于核矩阵:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# 将L2正则化系数为0.01的线性层应用于偏置向量：
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# 一个内核初始化为随机正交矩阵的线性层：
layers.Dense(64, kernel_initializer='orthogonal')

# 偏置矢量初始化为2.0s的线性层：
layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))
```

## 3. 训练和评估

### 3.1. 设置训练流程

构建好模型后，通过调用 `compile` 方法配置该模型的学习流程：

```python
model = tf.keras.Sequential([
# 向模型添加一个64单元的密集连接层：
layers.Dense(64, activation='relu', input_shape=(32,)),
# 加上另一个:
layers.Dense(64, activation='relu'),
# 添加具有10个输出单位的softmax层:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

`tf.keras.Model.compile` 采用三个重要参数：

* `optimizer`：此对象会指定训练过程。从`tf.keras.optimizers`  模块向其传递优化器实例，例如  `tf.keras.optimizers.Adam` 、`tf.keras.optimizers.SGD` 。如果您只想使用默认参数，还可以通过字符串指定优化器，例如'adam'或'sgd'。

* `loss`：要在优化期间最小化的函数。常见选择包括均方误差 (`mse`)、`categorical_crossentropy` 和 `binary_crossentropy`。损失函数由名称或通过从 `tf.keras.losses` 模块传递可调用对象来指定。

* `metrics`：用于监控训练。它们是 `tf.keras.metrics` 模块中的字符串名称或可调用对象。

* 此外，为了确保模型能够热切地进行训练和评估，您可以确保将`run_eagerly=True` 作为参数进行编译。

以下代码展示了配置模型以进行训练的几个示例：

```python
# 配置均方误差回归模型。
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # 均方误差
              metrics=['mae'])  # 平均绝对误差

# 为分类分类配置一个模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
```

### 3.2. 输入 NumPy 数据

对于小型数据集，请使用内存中的[NumPy](https://www.numpy.org/)数组训练和评估模型。使用 fit 方法使模型与训练数据“拟合”：

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
```

```
      ...
      Epoch 10/10
      1000/1000 [==============================] - 0s 82us/sample - loss: 11.4075 - categorical_accuracy: 0.1690
```

`tf.keras.Model.fit` 采用三个重要参数：

* `epochs`：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。

* `batch_size`：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。

* `validation_data`：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。

下面是使用 `validation_data` 的示例：

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

```
      Train on 1000 samples, validate on 100 samples
      ...
            Epoch 10/10
            1000/1000 [==============================] - 0s 93us/sample - loss: 11.5019 - categorical_accuracy: 0.1220 - val_loss: 11.5879 - val_categorical_accuracy: 0.0800
            <tensorflow.python.keras.callbacks.History at 0x7fe0642970b8>
```


### 3.3. 输入 tf.data 数据集

使用 [Datasets API](https://tensorflow.google.cn/guide/datasets) 可扩展为大型数据集或多设备训练。将 `tf.data.Dataset` 实例传递到 `fit` 方法：

```python
# 实例化玩具数据集实例：
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

# 在数据集上调用`fit`时，不要忘记指定`steps_per_epoch`。
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

输出：
```
      Epoch 1/10
      30/30 [==============================] - 0s 7ms/step - loss: 11.4902 - categorical_accuracy: 0.1094
```

在上方代码中，`fit` 方法使用了 `steps_per_epoch` 参数（表示模型在进入下一个周期之前运行的训练步数）。由于 `Dataset` 会生成批次数据，因此该代码段不需要 `batch_size`。

数据集也可用于验证：

```python
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10,
          validation_data=val_dataset)
```

```
      ...
      Epoch 10/10
      32/32 [==============================] - 0s 4ms/step - loss: 11.4778 - categorical_accuracy: 0.1560 - val_loss: 11.6653 - val_categorical_accuracy: 0.1300

      <tensorflow.python.keras.callbacks.History at 0x7fdfd8329d30>
```

### 3.4. 评估和预测

`tf.keras.Model.evaluate`和`tf.keras.Model.predict`方法可以使用NumPy数据和`tf.data.Dataset`。

要评估所提供数据的推理模式损失和指标，请运行以下代码：

```python
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
```

```
      1000/1000 [==============================] - 0s 72us/sample - loss: 11.5580 - categorical_accuracy: 0.0960
      30/30 [==============================] - 0s 2ms/step - loss: 11.4651 - categorical_accuracy: 0.1594

      [11.465100129445394, 0.159375]
```

要在所提供数据（采用 NumPy 数组形式）的推理中预测最后一层的输出，请运行以下代码：

```python
result = model.predict(data, batch_size=32)
print(result.shape)
```

```
      (1000, 10)
```

有关训练和评估的完整指南，包括如何从头开始编写自定义训练循环，请参阅[训练和评估指南](https://tensorflow.google.cn/alpha/guide/keras/training_and_evaluation)。

## 4. 构建高级模型

### 4.1. 函数式 API

`tf.keras.Sequential` 模型是层的简单堆叠，无法表示任意模型。使用 [Keras 函数式 API](https://tensorflow.google.cn/alpha/guide/keras/functional) 可以构建复杂的模型拓扑，例如：

* 多输入模型，
* 多输出模型，
* 具有共享层的模型（同一层被调用多次），
* 具有非序列数据流的模型（例如，剩余连接）。

使用函数式 API 构建的模型具有以下特征：

1. 层实例可调用并返回张量。
2. 输入张量和输出张量用于定义 `tf.keras.Model` 实例。
3. 此模型的训练方式和 `Sequential` 模型一样。

以下示例使用函数式 API 构建一个简单的全连接网络：

```python
inputs = tf.keras.Input(shape=(32,))  # 返回输入占位符

# 层实例可在张量上调用，并返回张量。
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
```

在给定输入和输出的情况下实例化模型。

```python
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# compile步骤指定训练配置
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练5个周期
model.fit(data, labels, batch_size=32, epochs=5)
```

```
      ...
      Epoch 5/5
      1000/1000 [==============================] - 0s 81us/sample - loss: 11.4819 - accuracy: 0.1270

      <tensorflow.python.keras.callbacks.History at 0x7fdfd820b898>
```

### 4.2. 模型子类化

通过对 `tf.keras.Model`进行子类化，并定义您自己的前向传播来构建完全可自定义的模型。在` __init__` 方法中创建层并将它们设置为类实例的属性。在 `call`方法中定义前向传播。

在启用 [eager execution](https://tensorflow.google.cn/alpha/guide/eager) 时，模型子类化特别有用，因为可以强制写入前向传播。

*注意：为了确保正向传递总是强制运行，你必须在调用超级构造函数时设置`dynamic = True`*

要点：针对作业使用正确的 API。虽然模型子类化较为灵活，但代价是复杂性更高且用户出错率更高。如果可能，请首选函数式 API。

以下示例展示了使用自定义前向传播进行子类化的 `tf.keras.Model`，该传递不必强制运行：

```python
class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # 在此处定义层。.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # 在这里定义你的前向传播
    # 使用之前定义的层（在`__init__`中）
    x = self.dense_1(inputs)
    return self.dense_2(x)
```

实例化新模型类：

```python
model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练5个周期
model.fit(data, labels, batch_size=32, epochs=5)
```

```
      ...
      Epoch 5/5 1000/1000 [==============================] - 0s 74us/sample - loss: 11.4954 - accuracy: 0.1110 
```

### 4.3. 自定义层

通过继承 `tf.keras.layers.Layer` 并实现以下方法来创建自定义层：

* `__init__`: （可选）定义此层要使用的子层

* `build`: 创建层的权重。使用 `add_weight` 方法添加权重。

* `call`: 定义前向传播。

* 或者，可以通过实现 `get_config` 方法和 `from_config` 类方法序列化层。

下面是一个自定义层的示例，它使用核矩阵实现输入的`matmul`：

```python
class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
```

使用自定义层创建模型：

```python
model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# 训练配置
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练5个周期
model.fit(data, labels, batch_size=32, epochs=5)
```

了解有关从头开始创建新层和模型的更多信息，在[从头开始编写层和模型指南](https://tensorflow.google.cn/alpha/guide/keras/custom_layers_and_models)。

## 5. 回调

回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。您可以编写自定义回调，也可以使用包含以下方法的内置 `tf.keras.callbacks`：


* `tf.keras.callbacks.ModelCheckpoint`: 定期保存模型的检查点。

* `tf.keras.callbacks.LearningRateScheduler`: 动态更改学习速率。

* `tf.keras.callbacks.EarlyStopping`:在验证效果不再改进时中断训练。

* `tf.keras.callbacks.TensorBoard`: 使用  [TensorBoard](https://tensorflow.google.cn/tensorboard) 监控模型的行为。

要使用  `tf.keras.callbacks.Callback`，请将其传递给模型的 `fit` 方法：

```python
callbacks = [
  # 如果`val_loss`在2个以上的周期内停止改进，则进行中断训练
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # 将TensorBoard日志写入`./logs`目录
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
```

```
      Train on 1000 samples, validate on 100 samples 
      ...
      Epoch 5/5 1000/1000 [==============================] - 0s 76us/sample - loss: 11.4813 - accuracy: 0.1190 - val_loss: 11.5753 - val_accuracy: 0.1100 <tensorflow.python.keras.callbacks.History at 0x7fdfd12e7080>
```


## 6. 保存和恢复

### 6.1. 仅限权重

使用 `tf.keras.Model.save_weights`保存并加载模型的权重：

```python
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```
# 将权重保存到TensorFlow检查点文件
model.save_weights('./weights/my_model')

# 恢复模型的状态，这需要具有相同架构的模型。
model.load_weights('./weights/my_model')
```

默认情况下，会以 [TensorFlow 检查点](https://tensorflow.google.cn/alpha/guide/checkpoints)文件格式保存模型的权重。权重也可以另存为 Keras HDF5 格式（Keras 多后端实现的默认格式）：

```
# 将权重保存到HDF5文件
model.save_weights('my_model.h5', save_format='h5')

# 恢复模型的状态
model.load_weights('my_model.h5')
```

### 6.2. 仅限配置

可以保存模型的配置，此操作会对模型架构（不含任何权重）进行序列化。即使没有定义原始模型的代码，保存的配置也可以重新创建并初始化相同的模型。Keras 支持 JSON 和 YAML 序列化格式：



```python
# 将模型序列化为JSON格式
json_string = model.to_json()
json_string
```

```
      '{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"units": 64, "activity_regularizer": null, "dtype": "float32",....... "backend": "tensorflow", "keras_version": "2.2.4-tf"}'
```

```python
import json
import pprint
pprint.pprint(json.loads(json_string))
```

```
      {'backend': 'tensorflow', 'class_name': 'Sequential', 'config': {'layers': [{'class_name': 'Dense', 'config': {'activation': 'relu', 'activity_regularizer': None, '......'keras_version': '2.2.4-tf'}
```

更多运行的输出内容请看英文版https://tensorflow.google.cn/alpha/guide/keras/overview

从 json 重新创建模型（刚刚初始化）。

```python
fresh_model = tf.keras.models.model_from_json(json_string)
```

将模型序列化为YAML格式，要求您在导入TensorFlow之前安装pyyaml（命令：`pip install -q pyyaml`）：

```
yaml_string = model.to_yaml()
print(yaml_string)
```

从YAML重新创建模型：

```python
fresh_model = tf.keras.models.model_from_yaml(yaml_string)
```

注意：子类化模型不可序列化，因为它们的架构由`call`方法正文中的 Python 代码定义。


### 6.3. 整个模型

整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以对模型设置检查点并稍后从完全相同的状态继续训练，而无需访问原始代码。

```python
# 创建一个简单的模型
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# 将整个模型保存到HDF5文件
model.save('my_model.h5')

# 重新创建完全相同的模型，包括权重和优化器
model = tf.keras.models.load_model('my_model.h5')
```

```
      ...
      Epoch 5/5 1000/1000 [==============================] - 0s 76us/sample - loss: 11.4913 - accuracy: 0.0990
```

 
在[保存和序列化模型指南](https://tensorflow.google.cn/alpha/guide/keras/saving_and_serializing)中，了解有关Keras模型的保存和序列化的更多信息。

## 7. Eager execution

[Eager execution](https://tensorflow.google.cn/guide/estimators) 是一种命令式编程环境，可立即评估操作。这不是Keras所必需的，但是由`tf.keras`支持，对于检查程序和调试很有用。

所有 `tf.keras` 模型构建 API 都与 Eager Execution 兼容。虽然可以使用 `Sequential` 和函数式 API，但 Eager Execution 对模型子类化和构建自定义层特别有用。与通过组合现有层来创建模型的 API 不同，函数式 API 要求您编写前向传播代码。

请参阅 [Eager Execution 指南](https://tensorflow.google.cn/guide/eager#build_a_model)，了解将 Keras 模型与自定义训练循环和 [tf.GradientTape](https://tensorflow.google.cn/api_docs/python/tf/GradientTape) 搭配使用的示例 [here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/advanced.ipynb).。

## 8. 分布


### 8.1. 多个 GPU

`tf.keras` 模型可以使用 `tf.distribute.Strategy`在多个 GPU 上运行。此 API 在多个 GPU 上提供分布式训练，几乎不需要更改现有代码。

目前，`tf.distribute.MirroredStrategy`是唯一受支持的分布策略。`MirroredStrategy` 通过在一台机器上使用规约在同步训练中进行图内复制。要使用`distribute.Strategy`s，请在 `Strategy`'s `.scope()`中嵌套优化器实例化和模型构造和编译，然后训练模型。


以下示例在单个计算机上的多个GPU之间分发`tf.keras.Model`。

首先，在分布式策略范围内定义模型：

```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
```

```
      Model: "sequential_5"
      _________________________________________________________________ 
      Layer (type) Output Shape Param # =================================================================
      dense_21 (Dense) (None, 16) 176 _________________________________________________________________ 
      dense_22 (Dense) (None, 1) 17 ================================================================= 
      Total params: 193 Trainable params: 193 Non-trainable params: 0   
      _________________________________________________________________
```

接下来，像往常一样训练模型数据：

```python
x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

model.fit(dataset, epochs=1)
```

```
32/32 [==============================] - 3s 82ms/step - loss: 0.7005 <tensorflow.python.keras.callbacks.History at 0x7fdfa057fb00>
```

有关更多信息，请参阅[TensorFlow中的分布式训练完整指南](https://tensorflow.google.cn/alpha/guide/distribute_strategy)。


> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-guide-keras-overview.html](https://www.mashangxue123.com/tensorflow/tf2-guide-keras-overview.html)
> 英文版本：[https://tensorflow.google.cn/alpha/guide/keras/overview](https://tensorflow.google.cn/alpha/guide/keras/overview)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/guide/keras/overview.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/guide/keras/overview.md)
