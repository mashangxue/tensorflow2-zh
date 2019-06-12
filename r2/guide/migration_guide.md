---
title: 将 TF1.x 代码迁移到 TensorFlow 2.0
categories: tensorflow2官方教程
tags: tensorflow2.0教程
top: 1903
abbrlink: tensorflow/tf2-guide-migration_guide
---

# 将 TF1.x 代码迁移到 TensorFlow 2.0（tensorflow2.0官方教程翻译）

在TensorFlow 2.0中，仍然可以运行未经修改的1.x代码（contrib除外）：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

但是，这并不能让您利用TensorFlow2.0中的许多改进。本指南将帮助您升级代码，使其更简单、更高效、更易于维护。

## 自动转换脚本

第一步是尝试运行[升级脚本](https://tensorflow.google.cn/beta/guide/upgrade).

这将在将您的代码升级到TensorFlow 2.0时执行初始步骤。但是它不能使您的代码适合TensorFlowF 2.0。您的代码仍然可以使用`tf.compat.v1` 接口来访问占位符，会话，集合和其他1.x样式的功能。

## 使代码2.0原生化


本指南将介绍将TensorFlow 1.x代码转换为TensorFlow 2.0的几个示例。这些更改将使您的代码利用性能优化和简化的API调用。
在每一种情况下，模式是：

### 1. 替换`tf.Session.run`调用

每个`tf.Session.run`调用都应该被Python函数替换。

* `feed_dict`和`tf.placeholder'成为函数参数。
* `fetches`成为函数的返回值。

您可以使用标准Python工具（如`pdb`）逐步调试和调试函数

如果您对它的工作感到满意，可以添加一个`tf.function`装饰器，使其在图形模式下高效运行。有关其工作原理的更多信息，请参阅[Autograph Guide](https://tensorflow.google.cn/beta/guide/autograph)。

### 2.  使用Python对象来跟踪变量和损失

使用`tf.Variable`而不是`tf.get_variable`。
每个`variable_scope`都可以转换为Python对象。通常这将是以下之一：

* `tf.keras.layers.Layer`
* `tf.keras.Model`
* `tf.Module`

如果需要聚合变量列表（如 `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)` ），请使用`Layer`和`Model`对象的`.variables`和`.trainable_variables`属性。

这些`Layer`和`Model`类实现了几个不需要全局集合的其他属性。他们的`.losses`属性可以替代使用`tf.GraphKeys.LOSSES`集合。

有关详细信息，请参阅[keras指南](https://tensorflow.google.cn/beta/guide/keras)。

警告：许多`tf.compat.v1`符号隐式使用全局集合。

### 3. 升级您的训练循环

使用适用于您的用例的最高级API。首选`tf.keras.Model.fit`构建自己的训练循环。

如果您编写自己的训练循环，这些高级函数可以管理很多可能容易遗漏的低级细节。例如，它们会自动收集正则化损失，并在调用模型时设置`training = True`参数。

### 4. 升级数据输入管道

使用`tf.data`数据集进行数据输入。这些对象是高效的，富有表现力的，并且与张量流很好地集成。

它们可以直接传递给`tf.keras.Model.fit`方法。

```python
model.fit(dataset, epochs=5)
```

它们可以直接在标准Python上迭代：

```python
for example_batch, label_batch in dataset:
    break
```


## 转换模型

### 设置


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import tensorflow_datasets as tfds
```

### 低阶变量和操作执行

低级API使用的示例包括：

* 使用变量范围来控制重用
* 用`tf.get_variable`创建变量。
* 显式访问集合
* 使用以下方法隐式访问集合：

  * `tf.global_variables`
  * `tf.losses.get_regularization_loss`

* 使用`tf.placeholder`设置图输入
* 用`session.run`执行图形
* 手动初始化变量


#### 转换前

以下是使用TensorFlow 1.x在代码中看起来像这些模式的内容：

```python
in_a = tf.placeholder(dtype=tf.float32, shape=(2))
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

def forward(x):
  with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)),
                        regularizer=tf.contrib.layers.l2_regularizer(0.04))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return W * x + b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss = tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss],
      	        feed_dict={in_a: [1, 0], in_b: [0, 1]})

```

#### 转换后

在转换后的代码中：

* 变量是本地Python对象.
* `forward`函数仍定义计算。
* `sess.run`调用被替换为对'forward`的调用
* 可以添加可选的`tf.function`装饰器以提高性能。
* 正则化是手动计算的，不涉及任何全局集合。
* **没有会话或占位符**


```python
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)
```


```python
out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss = regularizer(W)
```

### 基于`tf.layers`的模型

`tf.layers`模块用于包含依赖于`tf.variable_scope`来定义和重用变量的层函数。

#### 转换前

```python
def model(x, training, scope='model'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.04))
    x = tf.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, 0.1, training=training)
    x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.layers.dense(x, 10, activation=tf.nn.softmax)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)
```

#### 转换后

* 简单的层堆栈可以整齐地放入 `tf.keras.Sequential` 中。  (对于更复杂的模型，请参见 *自定义层和模型* ，以及 *函数式API* 两个教程）
* 模型跟踪变量和正则化损失
* 转换是一对一的，因为有一个从`tf.layers`到`tf.keras.layers`的直接映射。

大多数参数保持不变，但注意区别：

* 训练参数在运行时由模型传递给每个层
* 原来模型函数的第一个参数（input `x` ）消失，这是因为层将构建模型与调用模型分开了。

同时也要注意：

* 如果你使用来自`tf.contrib`的初始化器的正则化器，它们的参数变化比其他变量更多。
* 代码不在写入集合，因此像 `tf.losses.get_regularization_loss` 这样的函数将不再返回这些值，这可能会破坏您的训练循环。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.04),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))
```


```python
train_out = model(train_data, training=True)
print(train_out)
```


```python
test_out = model(test_data, training=False)
print(test_out)
```


```python
# 以下是所有可训练的变量。
len(model.trainable_variables)
```


```python
# 这是正规化损失。
model.losses
```

### 混合变量和tf.layers

现存的代码通常将较低级别的TF 1.x变量和操作与较高级的 `tf.layers` 混合。

#### 转换前
```python
def model(x, training, scope='model'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    W = tf.get_variable(
      "W", dtype=tf.float32,
      initializer=tf.ones(shape=x.shape),
      regularizer=tf.contrib.layers.l2_regularizer(0.04),
      trainable=True)
    if training:
      x = x + W
    else:
      x = x + W * 0.5
    x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.layers.flatten(x)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)
```

#### 转换后

要转换此代码，请遵循将图层映射到图层的模式，如上例所示。

一般模式是：

* 在`__init__`中收集图层参数。
* 在`build`中构建变量。
* 在`call`中执行计算，并返回结果。

`tf.variable_scope`实际上是它自己的一层。所以把它重写为`tf.keras.layers.Layer`。
有关信息请参阅 [指南](https://tensorflow.google.cn/beta/guide/keras/custom_layers_and_models) 

```python
# Create a custom layer for part of the model
class CustomLayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(CustomLayer, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=input_shape[1:],
        dtype=tf.float32,
        initializer=tf.keras.initializers.ones(),
        regularizer=tf.keras.regularizers.l2(0.02),
        trainable=True)

  # 调用方法有时会在图形模式下使用，训练会变成一个张量 
  @tf.function
  def call(self, inputs, training=None):
    if training:
      return inputs + self.w
    else:
      return inputs + self.w * 0.5
```


```python
custom_layer = CustomLayer()
print(custom_layer([1]).numpy())
print(custom_layer([1], training=True).numpy())
```


```python
train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

# 构建包含自定义层的模型 
model = tf.keras.Sequential([
    CustomLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
])

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

```

需要注意以下几点：

* 子类化的Keras模型和层需要在v1图(没有自动控制依赖关系)和eager模式下运行

* 将`call（）`包装在`tf.function（）`中以获取自动图和自动控制依赖关系

* 不要忘了调用时需要一个训练参数（ `tf.Tensor` 或Python布尔值）

* 使用`self.add_weight（）`在构造函数或`def build（）`中创建模型变量
  * 在`build`中，您可以访问输入形状，因此可以创建具有匹配形状的权重。
  * 使用`tf.keras.layers.Layer.add_weight`允许Keras跟踪变量和正则化损失。

* 不要在对象中保留`tf.Tensors`。
  * 它们可能在`tf.function`中或在 eager 的上下文中创建，并且这些张量的行为也不同。
  * 使用`tf.Variable`s作为状态，它们总是可用于两种情况
  * `tf.Tensors`仅适用于中间值。

### 关于Slim＆contrib.layers的说明

大量较旧的TensorFlow 1.x代码使用 [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) 库，与TensorFlow 1.x一起打包为`tf.contrib.layers`。作为`contrib`模块，TensorFlow 2.0中不再提供此功能，即使在`tf.compat.v1`中也是如此。使用Slim转换为TF 2.0比转换使用`tf.layers`的存储库更复杂。事实上，首先将Slim代码转换为`tf.layers`然后转换为Keras可能是有意义的。

- 删除 `arg_scopes`，所有args都需要显式

- 如果您使用它们，请将 `normalizer_fn `和 `activation_fn` 拆分为它们自己的图层

- 可分离的转换层映射到一个或多个不同的Keras层（深度、点和可分离的Keras层）

- Slim和 `tf.layers` 具有不同的arg名称和默认值

- 有些args有不同的尺度

- 如果您使用Slim预训练模型，请尝试使用 `tf.keras.applications` 或 [TFHub](https://tensorflow.orb/hub)

一些`tf.contrib`图层可能没有被移动到核心TensorFlow，而是被移动到了 [TF附加组件包](https://github.com/tensorflow/addons).


## 训练

有很多方法可以将数据提供给`tf.keras`模型。他们将接受Python生成器和Numpy数组作为输入。

将数据提供给模型的推荐方法是使用`tf.data`包，其中包含一组用于处理数据的高性能类。

如果您仍在使用tf.queue，则仅支持这些作为数据结构，而不是数据管道。

### 使用Datasets

[TensorFlow数据集包](https://tensorflow.org/datasets)  (`tfds`) 包含用于将预定义数据集加载为 `tf.data.Dataset`  对象的使用程序。

对于此示例，使用 `tfds` 加载MNIST数据集：

```python
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']
```

然后为训练准备数据：

  * 重新缩放每个图像
  * 打乱样本数据的顺序
  * 收集批量图像和标签



```python
BUFFER_SIZE = 10 # 实际代码中使用更大的值 
BATCH_SIZE = 64
NUM_EPOCHS = 5


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label
```

要使示例保持简短，请修剪数据集以仅返回5个批次：

```python
train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).take(5)
test_data = mnist_test.map(scale).batch(BATCH_SIZE).take(5)

STEPS_PER_EPOCH = 5

train_data = train_data.take(STEPS_PER_EPOCH)
test_data = test_data.take(STEPS_PER_EPOCH)
```


```
image_batch, label_batch = next(iter(train_data))
```

### 使用Keras训练循环

如果你不需要对训练过程进行低级别的控制，建议使用Keras内置的fit、evaluate和predict方法，这些方法提供了一个统一的接口来训练模型，而不管实现是什么（sequential、functional或子类化的）。

这些方法的有点包括：

-   它们接受Numpy数组、Python生成器和 `tf.data.Datasets`

-   它们自动应用正则化和激活损失

-   它们支持用于多设备训练的 `tf.distribute`

-   它们支持任意的callables作为损失和指标

-   它们支持回调，如 `tf.keras.callbacks.TensorBoard` 和自定义回调

-   它们具有高性能，可自动使用TensorFlow图形

以下是使用数据集训练模型的示例：

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 模型是没有自定义图层的完整模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=NUM_EPOCHS)
loss, acc = model.evaluate(test_data)

print("Loss {}, Accuracy {}".format(loss, acc))
```

### 编写你自己的训练循环

如果Keras模型的训练步骤适合您，但您需要在该步骤之外进行更多的控制，请考虑在您自己的数据迭代循环中使用  `tf.keras.model.train_on_batch` 方法。

记住：许多东西可以作为 `tf.keras.Callback` 的实现。

此方法具有上一节中提到的方法的许多优点，但允许用户控制外循环。

您还可以使用 `tf.keras.model.test_on_batch` 或 `tf.keras.Model.evaluate` 来检查训练期间的性能。

注意：`train_on_batch`和`test_on_batch`，默认返回单批的损失和指标。如果你传递`reset_metrics = False`，它们会返回累积的指标，你必须记住适当地重置指标累加器。还要记住，像 `AUC` 这样的一些指标需要正确计算 `reset_metrics = False`。

继续训练上面的模型：

```python
# 模型是没有自定义图层的完整模型 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

metrics_names = model.metrics_names

for epoch in range(NUM_EPOCHS):
  #Reset the metric accumulators
  model.reset_metrics()

  for image_batch, label_batch in train_data:
    result = model.train_on_batch(image_batch, label_batch)
    print("train: ",
          "{}: {:.3f}".format(metrics_names[0], result[0]),
          "{}: {:.3f}".format(metrics_names[1], result[1]))
  for image_batch, label_batch in test_data:
    result = model.test_on_batch(image_batch, label_batch,
                                 # return accumulated metrics
                                 reset_metrics=False)
  print("\neval: ",
        "{}: {:.3f}".format(metrics_names[0], result[0]),
        "{}: {:.3f}".format(metrics_names[1], result[1]))


```

<p id="custom_loops"/>

### 自定义训练步骤

如果您需要更多的灵活性和控制，可以通过实现自己的训练循环来实现，有三个步骤：

1. 迭代Python生成器或tf.data.Dataset以获取样本数据；

2. 使用tf.GradientTape收集渐变；

3. 使用tf.keras.optimizer将权重更新应用于模型。

记住：

-  始终在子类层和模型的调用方法中包含一个训练参数。

-  确保在正确设置训练参数的情况下调用模型。

-  根据使用情况，在对一批数据运行模型之前，模型变量可能不存在。

-  您需要手动处理模型的正则化损失等事情

请注意相对于v1的简化：

-  不需要运行变量初始化器，变量在创建时初始化。

-  不需要添加手动控制依赖项，即使在tf.function中，操作也像在eager模式下一样。

上面的模型：

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss = tf.math.add_n(model.losses)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(NUM_EPOCHS):
  for inputs, labels in train_data:
    train_step(inputs, labels)
  print("Finished epoch", epoch)

```

### 新型指标

在TensorFlow 2.0中，metrics是对象，Metrics对象在eager和tf.functions中运行，一个metrics具有以下方法：

* ` update_state()` – 添加新的观察结果

* `result()` – 给定观察值，获取metrics的当前结果

* `reset_states()` – 清除所有观察值

对象本身是可调用的，与 `update_state` 一样，调用新观察更新状态，并返回metrics的新结果。

你不需要手动初始化metrics的变量，而且因为TensorFlow 2.0具有自动控制依赖项，所以您也不需要担心这些。

下面的代码使用metrics来跟踪自定义训练循环中观察到的平均损失：

```python
# 创建metrics
loss_metric = tf.keras.metrics.Mean(name='train_loss')
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss = tf.math.add_n(model.losses)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # 更新metrics
  loss_metric.update_state(total_loss)
  accuracy_metric.update_state(labels, predictions)


for epoch in range(NUM_EPOCHS):
  # 重置metrics 
  loss_metric.reset_states()
  accuracy_metric.reset_states()

  for inputs, labels in train_data:
    train_step(inputs, labels)
  # 获取metric结果 
  mean_loss = loss_metric.result()
  mean_accuracy = accuracy_metric.result()

  print('Epoch: ', epoch)
  print('  loss:     {:.3f}'.format(mean_loss))
  print('  accuracy: {:.3f}'.format(mean_accuracy))

```

## 保存和加载


### Checkpoint兼容性

TensorFlow 2.0使用基于对象的检查点。

如果小心的话，仍然可以加载旧式的基于名称的检查点，代码转换过程可能会导致变量名的更改，但是有一些变通的方法。

最简单的方法是将新模型的名称与检查点的名称对齐：

-   变量仍然都有你可以设置的名称参数。

-   Keras模型还采用名称参数，并将其设置为变量的前缀。

-   `tf.name_scope` 函数可用于设置变量名称前缀，这与 `tf.variable_scope` 非常不同，它只影响名称，不跟踪变量和重用。

如果这不适合您的用例，请尝试使用 `tf.compat.v1.train.init_from_checkpoint` 函数，它需要一个 `assignment_map` 参数，该参数指定从旧名称到新名称的映射。

注意：与基于对象的检查点（可以[延迟加载](https://tensorflow.google.cn/beta/guide/checkpoints#loading_mechanics)不同，基于名称的检查点要求在调用函数时构建所有变量。某些模型推迟构建变量，直到您调用 `build` 或在一批数据上运行模型。

### 保存的模型兼容性

对于保存的模型没有明显的兼容性问题：

-   TensorFlow 1.x saved_models在TensorFlow 2.0中工作。

-   如果支持所有操作，TensorFlow 2.0 saved_models甚至可以在TensorFlow
    1.x中加载工作。

## Estimators

### 使用Estimators进行训练

TensorFlow 2.0支持Estimators，使用Estimators时，可以使用TensorFlow
1.x中的 `input_fn()` 、`tf.extimatro.TrainSpec` 和 `tf.estimator.EvalSpec`。

以下是使用 `input_fn` 和train以及evaluate的示例：

#### 创建input_fn和train/eval规范

```python
# 定义一个estimator的input_fn 
def input_fn():
  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000
  BATCH_SIZE = 64

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label[..., tf.newaxis]

  train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  return train_data.repeat()

# 定义 train & eval specs
train_spec = tf.estimator.TrainSpec(input_fn=input_fn,
                                    max_steps=STEPS_PER_EPOCH * NUM_EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn,
                                  steps=STEPS_PER_EPOCH)

```

### 使用Keras模型定义

在TensorFlow2.0中如何构建estimators存在一些差异。

我们建议您使用Keras定义模型，然后使用 `tf.keras.model_to_estimator` 将您的模型转换为estimator。下面的代码展示了如何在创建和训练estimator时使用这个功能。

```python
def make_model():
  return tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
```


```
model = make_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(
  keras_model = model
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

### 使用自定义 `model_fn`

如果您需要维护现有的自定义估算器 `model_fn`，则可以将 `model_fn` 转换为使用Keras模型。

但是出于兼容性原因，自定义 `model_fn` 仍将以1.x样式的图形模式运行，这意味着没有eager execution，也没有自动控制依赖。

在自定义 `model_fn` 中使用Keras模型类似于在自定义训练循环中使用它：

-  根据mode参数适当设置训练阶段

-  将模型的 `trainable_variables` 显示传递给优化器

但相对于自定义循环，存在重要差异：

-  使用 `tf.keras.Model.get_losses_for` 提取损失，而不是使用 `model.losses`

-  使用 `tf.keras.Model.get_updates_for` 提取模型的更新

注意：“更新”是每批后需要应用于模型的更改。例如，`tf.keras.layers.BatchNormalization`层中均值和方差的移动平均值。

以下代码从自定义`model_fn`创建一个估算器，说明所有这些问题。

```python
def my_model_fn(features, labels, mode):
  model = make_model()

  optimizer = tf.compat.v1.train.AdamOptimizer()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  training = (mode == tf.estimator.ModeKeys.TRAIN)
  predictions = model(features, training=training)

  reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
  total_loss = loss_fn(labels, predictions) + tf.math.add_n(reg_losses)

  accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                           predictions=tf.math.argmax(predictions, axis=1),
                                           name='acc_op')

  update_ops = model.get_updates_for(None) + model.get_updates_for(features)
  minimize_op = optimizer.minimize(
      total_loss,
      var_list=model.trainable_variables,
      global_step=tf.compat.v1.train.get_or_create_global_step())
  train_op = tf.group(minimize_op, update_ops)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=total_loss,
    train_op=train_op, eval_metric_ops={'accuracy': accuracy})

# Create the Estimator & Train
estimator = tf.estimator.Estimator(model_fn=my_model_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

## TensorShape

这个类被简化为保存`int`s，而不是`tf.compat.v1.Dimension`对象。所以不需要调用`.value（）`来获得`int`。

仍然可以从`tf.TensorShape.dims`访问单个`tf.compat.v1.Dimension`对象。

以下演示了TensorFlow 1.x和TensorFlow 2.0之间的区别。

```
# 创建一个shape并选择一个索引 
i = 0
shape = tf.TensorShape([16, None, 256])
shape
```

TF 1.x 运行:

```python
value = shape[i].value
```

TF 2.0 运行::



```python
value = shape[i]
value
```

 TF 1.x 运行::

```python
for dim in shape:
    value = dim.value
    print(value)
```

TF 2.0 运行::


```python
for value in shape:
  print(value)
```

在TF 1.x（或使用任何其他维度方法）中运行：

```python
dim = shape[i]
dim.assert_is_compatible_with(other_dim)
```

TF 2.0运行：


```python
other_dim = 16
Dimension = tf.compat.v1.Dimension

if shape.rank is None:
  dim = Dimension(None)
else:
  dim = shape.dims[i]
dim.is_compatible_with(other_dim) # or any other dimension method
```


```python
shape = tf.TensorShape(None)

if shape:
  dim = shape.dims[i]
  dim.is_compatible_with(other_dim) # or any other dimension method
```

如果等级已知，则 `tf.TensorShape` 的布尔值为“True”，否则为“False”。

```python
print(bool(tf.TensorShape([])))      # 标量 Scalar 
print(bool(tf.TensorShape([0])))     # 0长度的向量 vector
print(bool(tf.TensorShape([1])))     # 1长度的向量 vector
print(bool(tf.TensorShape([None])))  # 未知长度的向量 
print(bool(tf.TensorShape([1, 10, 100])))       # 3D tensor
print(bool(tf.TensorShape([None, None, None]))) # 3D tensor with no known dimensions
print()
print(bool(tf.TensorShape(None)))  # 未知等级的张量 
```

## 其他行为改变

您可能会遇到TensorFlow 2.0中的一些其他行为变化。


### ResourceVariables

TensorFlow 2.0默认创建`ResourceVariables`，而不是`RefVariables`。

`ResourceVariables`被锁定用于写入，因此提供更直观的一致性保证。

* 这可能会改变边缘情况下的行为
* 这可能偶尔会创建额外的副本，可能会有更高的内存使用量
* 可以通过将`use_resource = False`传递给`tf.Variable`构造函数来禁用它。

### Control Flow

控制流op实现得到了简化，因此在TensorFlow 2.0中生成了不同的图。

## 结论

回顾一下本节内容:

1. 运行更新脚本
2. 删除contrib符号
3. 将模型切换为面向对象的样式（Keras）
4. 尽可能使用`tf.keras`或`tf.estimator`培训和评估循环。
5. 否则，请使用自定义循环，但请务必避免会话和集合。

将代码转换为TensorFlow 2.0需要一些工作，但会有以下改变：
-   更少的代码行
-   提高清晰度和简洁性
-   调试更简单

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-guide-migration_guide.html](https://www.mashangxue123.com/tensorflow/tf2-guide-migration_guide.html)
> 英文版本：[https://tensorflow.google.cn/beta/guide/migration_guide](https://tensorflow.google.cn/beta/guide/migration_guide)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/guide/migration_guide.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/guide/migration_guide.md)
