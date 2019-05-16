
# Keras：概述

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/guide/keras/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/guide/keras/overview.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/guide/keras/overview.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

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

安装pyyaml （可选）：`pip install -q pyyaml`


`tf.keras` can run any Keras-compatible code, but keep in mind:

* The `tf.keras` version in the latest TensorFlow release might not be the same
  as the latest `keras` version from PyPI. Check `tf.keras.__version__`.
* When [saving a model's weights](#weights_only), `tf.keras` defaults to the
  [checkpoint format](./checkpoints.md). Pass `save_format='h5'` to
  use HDF5.

`tf.keras` 可以运行任何与 Keras 兼容的代码，但请注意：

* 最新版 TensorFlow 中的 `tf.keras` 版本可能与 PyPI 中的最新 keras 版本不同。请查看 `tf.keras.version`。

* [保存模型的权重](#weights_only)时，`tf.keras` 默认采用检查点格式。请传递 ` save_format='h5' `以使用 HDF5。

## 2. 构建简单的模型

### 2.1. 序列模型

在 Keras 中，您可以通过组合层来构建模型。模型（通常）是由层构成的图。最常见的模型类型是层的堆叠：`tf.keras.Sequential` 模型。

要构建一个简单的全连接网络（即多层感知器），请运行以下代码：

```
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

```
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

```
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

```
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

```
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

```
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

Use the [Datasets API](./datasets.md) to scale to large datasets
or multi-device training. Pass a `tf.data.Dataset` instance to the `fit`
method:

使用 Datasets API 可扩展为大型数据集或多设备训练。将 tf.data.Dataset 实例传递到 fit 方法：

```
# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

Here, the `fit` method uses the `steps_per_epoch` argument—this is the number of
training steps the model runs before it moves to the next epoch. Since the
`Dataset` yields batches of data, this snippet does not require a `batch_size`.

Datasets can also be used for validation:


```
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10,
          validation_data=val_dataset)
```

### 3.4. Evaluate and predict

The `tf.keras.Model.evaluate` and `tf.keras.Model.predict` methods can use NumPy
data and a `tf.data.Dataset`.

To *evaluate* the inference-mode loss and metrics for the data provided:


```
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
```

And to *predict* the output of the last layer in inference for the data provided,
as a NumPy array:


```
result = model.predict(data, batch_size=32)
print(result.shape)
```

For a complete guide on training and evaluation, including how to write custom training loops from sratch, see the [Guide to Training & Evaluation](./training_and_evaluation.ipynb).

## 4. Build advanced models

### 4.1. Functional API

 The `tf.keras.Sequential` model is a simple stack of layers that cannot
represent arbitrary models. Use the
[Keras functional API](./functional.ipynb)
to build complex model topologies such as:

* Multi-input models,
* Multi-output models,
* Models with shared layers (the same layer called several times),
* Models with non-sequential data flows (e.g. residual connections).

Building a model with the functional API works like this:

1. A layer instance is callable and returns a tensor.
2. Input tensors and output tensors are used to define a `tf.keras.Model`
   instance.
3. This model is trained just like the `Sequential` model.

The following example uses the functional API to build a simple, fully-connected
network:


```
inputs = tf.keras.Input(shape=(32,))  # Returns an input placeholder

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
```

Instantiate the model given inputs and outputs.


```
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
```

### 4.2. Model subclassing

Build a fully-customizable model by subclassing `tf.keras.Model` and defining
your own forward pass. Create layers in the `__init__` method and set them as
attributes of the class instance. Define the forward pass in the `call` method.

Model subclassing is particularly useful when
[eager execution](./eager.md) is enabled since the forward pass
can be written imperatively.

Note: to make sure the forward pass is *always* run imperatively, you must set `dynamic=True` when calling the super constructor.

Key Point: Use the right API for the job. While model subclassing offers
flexibility, it comes at a cost of greater complexity and more opportunities for
user errors. If possible, prefer the functional API.

The following example shows a subclassed `tf.keras.Model` using a custom forward
pass that does not have to be run imperatively:


```
class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)
```

Instantiate the new model class:


```
model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

### 4.3. Custom layers

Create a custom layer by subclassing `tf.keras.layers.Layer` and implementing
the following methods:

* `__init__`: Optionally define sublayers to be used by this layer.
* `build`: Create the weights of the layer. Add weights with the `add_weight`
  method.
* `call`: Define the forward pass.
* Optionally, a layer can be serialized by implementing the `get_config` method
  and the `from_config` class method.

Here's an example of a custom layer that implements a `matmul` of an input with
a kernel matrix:


```
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

Create a model using your custom layer:


```
model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

Learn more about creating new layers and models from scratch with subclassing in the [Guide to writing layers and models from scratch](./custom_layers_and_models.ipynb).

## 5. Callbacks

A callback is an object passed to a model to customize and extend its behavior
during training. You can write your own custom callback, or use the built-in
`tf.keras.callbacks` that include:

* `tf.keras.callbacks.ModelCheckpoint`: Save checkpoints of your model at
  regular intervals.
* `tf.keras.callbacks.LearningRateScheduler`: Dynamically change the learning
  rate.
* `tf.keras.callbacks.EarlyStopping`: Interrupt training when validation
  performance has stopped improving.
* `tf.keras.callbacks.TensorBoard`: Monitor the model's behavior using
  [TensorBoard](https://tensorflow.org/tensorboard).

To use a `tf.keras.callbacks.Callback`, pass it to the model's `fit` method:


```
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
```


## 6. Save and restore

### 6.1. Weights only

Save and load the weights of a model using `tf.keras.Model.save_weights`:


```
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')
```

By default, this saves the model's weights in the
[TensorFlow checkpoint](../checkpoints.md) file format. Weights can
also be saved to the Keras HDF5 format (the default for the multi-backend
implementation of Keras):


```
# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')
```

### 6.2. Configuration only

A model's configuration can be saved—this serializes the model architecture
without any weights. A saved configuration can recreate and initialize the same
model, even without the code that defined the original model. Keras supports
JSON and YAML serialization formats:


```
# Serialize a model to JSON format
json_string = model.to_json()
json_string
```


```
import json
import pprint
pprint.pprint(json.loads(json_string))
```

Recreate the model (newly initialized) from the JSON:


```
fresh_model = tf.keras.models.model_from_json(json_string)
```

Serializing a model to YAML format requires that you install `pyyaml` *before you import TensorFlow*:


```
yaml_string = model.to_yaml()
print(yaml_string)
```

Recreate the model from the YAML:


```
fresh_model = tf.keras.models.model_from_yaml(yaml_string)
```

Caution: Subclassed models are not serializable because their architecture is
defined by the Python code in the body of the `call` method.


### 6.3. Entire model

The entire model can be saved to a file that contains the weight values, the
model's configuration, and even the optimizer's configuration. This allows you
to checkpoint a model and resume training later—from the exact same
state—without access to the original code.


```
# Create a simple model
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')
```

Learn more about saving and serialization for Keras models in the [Guide to saving and Serializing Models](./saving_and_serializing.ipynb).

## 7. Eager execution

[Eager execution](./eager.md) is an imperative programming
environment that evaluates operations immediately. This is not required for
Keras, but is supported by `tf.keras` and useful for inspecting your program and
debugging.

All of the `tf.keras` model-building APIs are compatible with eager execution.
And while the `Sequential` and functional APIs can be used, eager execution
especially benefits *model subclassing* and building *custom layers*—the APIs
that require you to write the forward pass as code (instead of the APIs that
create models by assembling existing layers).

See the [eager execution guide](./eager.ipynb#build_a_model) for
examples of using Keras models with custom training loops and `tf.GradientTape`.
You can also find a complete, short example [here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/advanced.ipynb).

## 8. Distribution


### 8.1. Multiple GPUs

`tf.keras` models can run on multiple GPUs using
`tf.distribute.Strategy`. This API provides distributed
training on multiple GPUs with almost no changes to existing code.

Currently, `tf.distribute.MirroredStrategy` is the only supported
distribution strategy. `MirroredStrategy` does in-graph replication with
synchronous training using all-reduce on a single machine. To use
`distribute.Strategy`s , nest the optimizer instantiation and model construction and compilation in a `Strategy`'s `.scope()`, then
train the model.

The following example distributes a `tf.keras.Model` across multiple GPUs on a
single machine.

First, define a model inside the distributed strategy scope:


```
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
```

Next, train the model on data as usual:


```
x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

model.fit(dataset, epochs=1)
```

For more information, see the [full guide on Distributed Training in TensorFlow](../distribute_strategy.ipynb).
