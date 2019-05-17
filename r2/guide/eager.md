---
title: Eager Execution 概述
tags: 
    - tensorflow2.0
categories: 
    - tensorflow2官方文档
date: 2019-05-10
abbrlink: tensorflow/tensorflow2-guide-eager
---

# Eager Execution 概述

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/guide/eager"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/guide/eager.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/guide/eager.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

TensorFlow 的 Eager Execution 是一种命令式编程环境，可立即评估操作，无需构建图：操作会返回具体的值，而不是构建以后再运行的计算图。这样能让您轻松地开始使用 TensorFlow 和调试模型，并且还减少了样板代码。要遵循本指南，请在交互式 python 解释器中运行下面的代码示例。

Eager Execution 是一个灵活的机器学习平台，用于研究和实验，可提供：

* *直观的界面* - 自然地组织代码结构并使用 Python 数据结构。快速迭代小模型和小型数据集。

* *更轻松的调试功能* - 直接调用操作以检查正在运行的模型并测试更改。使用标准 Python 调试工具进行即时错误报告。

* *自然控制流程* - 使用 Python 控制流程而不是图控制流程，简化了动态模型的规范。

Eager Execution 支持大多数 TensorFlow 操作和 GPU 加速。

注意：如果启用 Eager Execution，某些模型的开销可能会增加。我们正在改进性能；如果发现问题，请报告错误，并分享您的基准测试结果。


## 设置和基本用法

升级到最新版本的 TensorFlow：

```python
from __future__ import absolute_import, division, print_function, unicode_literals

# pip install tensorflow==2.0.0-alpha0
import tensorflow as tf
```

在Tensorflow 2.0中，默认情况下启用了Eager Execution。

```python
tf.executing_eagerly()
```

```
      True
```

现在您可以运行TensorFlow操作，结果将立即返回：

```python
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
```

```
      hello, [[4.]]
```

启用 Eager Execution 会改变 TensorFlow 操作的行为方式(现在它们会立即评估并将值返回给 Python)。`tf.Tensor` 对象会引用具体值，而不是指向计算图中的节点的符号句柄。由于不需要构建稍后在会话中运行的计算图，因此使用 `print()` 或调试程序很容易检查结果。评估、输出和检查张量值不会中断计算梯度的流程。

Eager Execution 适合与 NumPy 一起使用。NumPy 操作接受`tf.Tensor` 参数。TensorFlow [数学运算](https://tensorflow.google.cn/api_guides/python/math_ops) 将 Python 对象和 NumPy 数组转换为 `tf.Tensor` 对象。`tf.Tensor.numpy` 方法返回对象的值作为 NumPy  `ndarray`。

```python
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
```

```
      tf.Tensor(
      [[1 2]
       [3 4]], shape=(2, 2), dtype=int32)
```


```python
# Broadcasting support
b = tf.add(a, 1)
print(b)
```

```
      tf.Tensor(
      [[2 3]
       [4 5]], shape=(2, 2), dtype=int32)
```

```python
# Operator overloading is supported
print(a * b)
```

```
      tf.Tensor(
      [[ 2  6]
       [12 20]], shape=(2, 2), dtype=int32)
```


```python
# 使用NumPy值
import numpy as np

c = np.multiply(a, b)
print(c)
```

```
      [[ 2  6]
       [12 20]]
```


```python
# 从张量中获取numpy值：
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

## 动态控制流

Eager Execution 的一个主要好处是，在执行模型时，主机语言的所有功能都可用。因此，编写 [fizzbuzz](https://baike.baidu.com/item/FizzBuzz%E9%97%AE%E9%A2%98/16083686?fr=aladdin)很容易（举例而言）：

*FizzBuzz问题：举个例子，编写一个程序从1到100.当遇到数字为3的倍数的时候，点击“Fizz”替代数字，5的倍数用“Buzz”代替，既是3的倍数又是5的倍数点击“FizzBuzz”。* 

```python
def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1
```

```python
fizzbuzz(15)
```

```
1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz
```

这段代码具有依赖于张量值的条件并在运行时输出这些值。


## 构建模型

许多机器学习模型通过组合层来表示。将 TensorFlow 与 Eager Execution 结合使用时，您可以编写自己的层或使用在 `tf.keras.layers` 程序包中提供的层。

虽然您可以使用任何 Python 对象表示层，但 TensorFlow 提供了便利的基类 `tf.keras.layers.Layer`。您可以通过继承它实现自己的层，如果必须强制执行该层，在构造函数中设置 `self.dynamic=True`：

```python
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units
    self.dynamic = True

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # 构建方法在第一次使用图层时被调用。
    # 在build()上创建变量允许您使其形状取决于输入形状，因此无需用户指定完整形状。 
    # 如果您已经知道它们的完整形状，则可以在` __init__()`期间创建变量。
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # 覆盖 `call()` 而不是`__call__`，这样我们就可以执行一些记帐。
    return tf.matmul(input, self.kernel)
```

请使用`tf.keras.layers.Dense`层（而不是上面的`MySimpleLayer`），因为它具有其功能的超集（它也可以添加偏差）。

将层组合成模型时，可以使用 `tf.keras.Sequential` 表示由层线性堆叠的模型。它非常适合用于基本模型：

```python
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])
```

或者，通过继承 `tf.keras.Model` 将模型整理为类。这是一个本身也是层的层容器，允许 `tf.keras.Model`对象包含其他  `tf.keras.Model` 对象。

```python
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()
```

因为第一次将输入传递给层时已经设置参数，所以不需要为`tf.keras.Model` 类设置输入形状。

`tf.keras.layers` 类会创建并包含自己的模型变量，这些变量与其层对象的生命周期相关联。要共享层变量，请共享其对象。

## Eager 训练

### 计算梯度

[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)对于实现机器学习算法（例如用于训练神经网络的[反向传播](https://en.wikipedia.org/wiki/Backpropagation)）来说很有用。在 Eager Execution 期间，请使用 `tf.GradientTape` 跟踪操作以便稍后计算梯度。

`tf.GradientTape`  是一种选择性功能，可在不跟踪时提供最佳性能。由于在每次调用期间都可能发生不同的操作，因此所有前向传播操作都会记录到“磁带”中。要计算梯度，请反向播放磁带，然后放弃。特定的 `tf.GradientTape`  只能计算一个梯度；随后的调用会抛出运行时错误。

```python
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```


### 训练模型

以下示例将创建一个多层模型，该模型会对标准 MNIST 手写数字进行分类。它演示了在 Eager Execution 环境中构建可训练图的优化器和层 API。

```python
# 获取并格式化mnist数据
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)
```


```python
# 建立模型
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
```

即使没有训练，也可以在 Eager Execution 中调用模型并检查输出：

```python
for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())
```

```
      Logits: [[-1.9521490e-02 2.2975644e-02 2.8935237e-02 2.0388789e-02 -1.8511273e-02 -6.4317137e-05 6.0662534e-03 -1.7174225e-02 5.4899108e-02 -2.8871424e-02]]
```

虽然 keras 模型具有内置训练循环（使用 `fit` 方法），但有时您需要更多自定义设置。下面是一个用 eager 实现的训练循环示例：

```python
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
```


```python
for (batch, (images, labels)) in enumerate(dataset.take(400)):
  if batch % 10 == 0:
    print('.', end='')
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)
    loss_value = loss_object(labels, logits)

  loss_history.append(loss_value.numpy().mean())
  grads = tape.gradient(loss_value, mnist_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
```


```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
```

```
      Text(0, 0.5, 'Loss [entropy]')
```

该示例使用了 [TensorFlow MNIST 示例](https://github.com/tensorflow/models/tree/master/official/mnist) 中的 [dataset.py](https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py) 模块，请将该文件下载到本地目录。运行以下命令以将 MNIST 数据文件下载到工作目录并准备要进行训练的 tf.data.Dataset：

### 变量和优化器

`tf.Variable` 对象会存储在训练期间访问的可变 `tf.Tensor` 值，以更加轻松地实现自动微分。模型的参数可以作为变量封装在类中。

通过将 `tf.Variable` 与 `tf.GradientTape` 结合使用可以更好地封装模型参数。例如，上面的自动微分示例可以重写为：

```python
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# 点数约为3 * x + 2的玩具数据集
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# 要优化的损失函数
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
```

## 在Eager Execution期间将对象用于状态

使用 TF 1.x的 Graph Execution 时，程序状态（如变量）存储在全局集合中，它们的生命周期由 `tf.Session` 对象管理。相反，在Eager Execution期间，状态对象的生命周期由其对应的 Python 对象的生命周期决定。

### 变量是对象

在 Eager Execution 期间，变量会一直存在，直到相应对象的最后一个引用被移除，然后变量被删除。

```python
if tf.test.is_gpu_available():
  with tf.device("gpu:0"):
    v = tf.Variable(tf.random.normal([1000, 1000]))
    v = None  # v no longer takes up GPU memory
```

### 基于对象的保存

本节是[训练检查点指南](https://tensorflow.google.cn/alpha/guide/checkpoints)的简短版本。

`tf.train.Checkpoint` 可以将 `tf.Variable` 保存到检查点并从中恢复：

```python
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
```

```
x.assign(2.)   # 为变量分配新值并保存。
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')
```

```python
x.assign(11.)  # 保存后更改变量。

# 从检查点恢复值
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0
```

要保存和加载模型，`tf.train.Checkpoint` 会存储对象的内部状态，而不需要隐藏变量。要记录 `model`、`optimizer` 和全局步的状态，请将它们传递到 `tf.train.Checkpoint`：

```python
import os

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint_dir = 'path/to/model_dir'
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)

root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

注意：在许多训练循环中，在调用`tf.train.Checkpoint.restore`之后创建变量。这些变量将在创建后立即恢复，并且可以使用断言来确保检查点已完全加载。有关详细信息，请参阅[训练检查点指南](https://tensorflow.google.cn/alpha/guide/checkpoints)。

### 面向对象的指标

`tf.keras.metrics`存储为对象。通过将新数据传递给可调用对象来更新指标，并使用  `tf.keras.metrics.result`方法检索结果，例如：

```python
m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```

## 自动微分高级内容

### 动态模型

`tf.GradientTape` 也可用于动态模型。这个回溯线搜索算法示例看起来像普通的 NumPy 代码，除了存在梯度并且可微分，尽管控制流比较复杂：

```python
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
```

### 自定义梯度

自定义梯度是一种覆盖梯度的简单方法。在正向函数中，定义相对于输入、输出或中间结果的梯度。例如，下面是在反向传播中截断梯度范数的一种简单方式：

```python
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

自定义梯度通常用于为一系列操作提供数值稳定的梯度：

```python
def log1pexp(x):
  return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

```


```python
# 梯度计算在x = 0时工作正常。
grad_log1pexp(tf.constant(0.)).numpy()   # => 0.5
```

`0.5`

```python
# 但是，由于数值不稳定，x = 100失败。
grad_log1pexp(tf.constant(100.)).numpy()  # => nan
```

`nan`

在此处，`log1pexp` 函数可以通过自定义梯度进行分析简化。下面的实现重用了在前向传播期间计算的`tf.exp(x)`的值，通过消除冗余计算，变得更加高效：

```python
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

```


```python
# 和以前一样，梯度计算在x = 0时工作正常。
grad_log1pexp(tf.constant(0.)).numpy()    # => 0.5
```


```python
# 并且梯度计算也适用于x = 100。
grad_log1pexp(tf.constant(100.)).numpy()   # => 1.0
```

## 性能

在Eager Execution期间，计算会自动分流到 GPU。如果要控制计算运行的位置，可以将其放在`tf.device('/gpu:0')`  块（或 CPU 等效块）中：

```python
import time

def measure(x, steps):
  # TensorFlow在第一次使用时初始化GPU，从计时中排除。
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul can return before completing the matrix multiplication
  # (e.g., can return after enqueing the operation on a CUDA stream).
  # The x.numpy() call below will ensure that all enqueued operations
  # have completed (and will also copy the result to host memory,
  # so we're including a little more than just the matmul operation
  # time).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# Run on GPU, if available:
if tf.test.is_gpu_available():
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
else:
  print("GPU: not found")
```

```
      Time to multiply a (1000, 1000) matrix by itself 200 times:
      CPU: 0.7741374969482422 secs
      GPU: not found
```

`tf.Tensor`对象可以复制到不同的设备来执行其操作：

```python
if tf.test.is_gpu_available():
  x = tf.random.normal([10, 10])

  x_gpu0 = x.gpu()
  x_cpu = x.cpu()

  _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
  _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

```

### 基准

对于计算量繁重的模型（如在 GPU 上训练的 [ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)），Eager Execution 性能与 `tf.function` Execution 相当。但是对于计算量较小的模型来说，这种性能差距会越来越大，并且有很多工作要做，以便为具有大量小操作的模型优化热代码路径。

## 使用`tf.function`

虽然Eager Execution使开发和调试更具交互性，但TensorFlow 1.x样式图执行在分布式训练，性能优化和生产部署方面具有优势。为了弥补这一差距，TensorFlow 2.0通过`tf.function` API引入此功能。有关更多信息，请参阅[Autograph指南](https://tensorflow.google.cn/alpha/guide/autograph)。

