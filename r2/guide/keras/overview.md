
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

安装pyyaml （可选）：`pip install -q pyyaml`

```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
```

`tf.keras` can run any Keras-compatible code, but keep in mind:

* The `tf.keras` version in the latest TensorFlow release might not be the same
  as the latest `keras` version from PyPI. Check `tf.keras.__version__`.
* When [saving a model's weights](#weights_only), `tf.keras` defaults to the
  [checkpoint format](./checkpoints.md). Pass `save_format='h5'` to
  use HDF5.

`tf.keras` 可以运行任何与 Keras 兼容的代码，但请注意：

* 最新版 TensorFlow 中的 `tf.keras` 版本可能与 PyPI 中的最新 keras 版本不同。请查看 `tf.keras.version`。
* [保存模型的权重](#weights_only)时，`tf.keras` 默认采用检查点格式。请传递 ` save_format='h5' `以使用 HDF5。

## 2. Build a simple model

### 2.1. Sequential model

In Keras, you assemble *layers* to build *models*. A model is (usually) a graph
of layers. The most common type of model is a stack of layers: the
`tf.keras.Sequential` model.

To build a simple, fully-connected network (i.e. multi-layer perceptron):


```
from tensorflow.keras import layers

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))
```

You can find a complete, short example of how to use Sequential models [here](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/beginner.ipynb).

To learn about building more advanced models than Sequential models, see:
- [Guide to the Keras Functional](./functional.ipynb)
- [Guide to writing layers and models from scratch with subclassing](./custom_layers_and_models.ipynb)

### 2.2. Configure the layers

There are many `tf.keras.layers` available with some common constructor
parameters:

* `activation`: Set the activation function for the layer. This parameter is
  specified by the name of a built-in function or as a callable object. By
  default, no activation is applied.
* `kernel_initializer` and `bias_initializer`: The initialization schemes
  that create the layer's weights (kernel and bias). This parameter is a name or
  a callable object. This defaults to the `"Glorot uniform"` initializer.
* `kernel_regularizer` and `bias_regularizer`: The regularization schemes
  that apply the layer's weights (kernel and bias), such as L1 or L2
  regularization. By default, no regularization is applied.

The following instantiates `tf.keras.layers.Dense` layers using constructor
arguments:


```
# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.keras.activations.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))
```

## 3. Train and evaluate

### 3.1. Set up training

After the model is constructed, configure its learning process by calling the
`compile` method:


```
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

`tf.keras.Model.compile` takes three important arguments:

* `optimizer`: This object specifies the training procedure. Pass it optimizer
  instances from the `tf.keras.optimizers` module, such as
  `tf.keras.optimizers.Adam` or
  `tf.keras.optimizers.SGD`. If you just want to use the default parameters, you can also specify optimizers via strings, such as `'adam'` or `'sgd'`.
* `loss`: The function to minimize during optimization. Common choices include
  mean square error (`mse`), `categorical_crossentropy`, and
  `binary_crossentropy`. Loss functions are specified by name or by
  passing a callable object from the `tf.keras.losses` module.
* `metrics`: Used to monitor training. These are string names or callables from
  the `tf.keras.metrics` module.
* Additionally, to make sure the model trains and evaluates eagerly, you can make sure to pass `run_eagerly=True` as a parameter to compile.


The following shows a few examples of configuring a model for training:


```
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
```

### 3.2. Train from NumPy data

For small datasets, use in-memory [NumPy](https://www.numpy.org/){:.external}
arrays to train and evaluate a model. The model is "fit" to the training data
using the `fit` method:


```
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
```

`tf.keras.Model.fit` takes three important arguments:

* `epochs`: Training is structured into *epochs*. An epoch is one iteration over
  the entire input data (this is done in smaller batches).
* `batch_size`: When passed NumPy data, the model slices the data into smaller
  batches and iterates over these batches during training. This integer
  specifies the size of each batch. Be aware that the last batch may be smaller
  if the total number of samples is not divisible by the batch size.
* `validation_data`: When prototyping a model, you want to easily monitor its
  performance on some validation data. Passing this argument—a tuple of inputs
  and labels—allows the model to display the loss and metrics in inference mode
  for the passed data, at the end of each epoch.

Here's an example using `validation_data`:


```
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

### 3.3. Train from tf.data datasets

Use the [Datasets API](./datasets.md) to scale to large datasets
or multi-device training. Pass a `tf.data.Dataset` instance to the `fit`
method:


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
