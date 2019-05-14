---
title: 高效的TensorFlow 2.0
tags: 
    - tensorflow2.0
categories: 
    - tensorflow2.0官方文档
date: 2019-05-10
abbrlink: tensorflow/tensorflow2-guide-effective_tf2
---

# 高效的TensorFlow 2.0

TensorFlow 2.0中有多处更改，以使TensorFlow用户使用更高效。TensorFlow 2.0删除[冗余 APIs](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md),使API更加一致([统一 RNNs](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md),[统一优化器](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)),并通过[Eager execution](https://www.tensorflow.org/guide/eager)模式更好地与Python运行时集成

许多[RFCs](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr)已经解释了TensorFlow 2.0所带来的变化。本指南介绍了TensorFlow 2.0应该是什么样的开发，假设您对TensorFlow 1.x有一定的了解。

## 1. 主要变化的简要总结

### 1.1. API清理

许多API在tensorflow 2.0中[消失或移动](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)。一些主要的变化包括删除`tf.app`、`tf.flags`和`tf.logging` ，转而支持现在开源的[absl-py](https://github.com/abseil/abseil-py)，重新安置`tf.contrib`中的项目，并清理主要的 `tf.*`命名空间，将不常用的函数移动到像 `tf.math`这样的子包中。一些API已被2.0版本等效替换，如`tf.summary`, `tf.keras.metrics`和`tf.keras.optimizers`。
自动应用这些重命名的最简单方法是使用[v2升级脚本](https://tensorflow.google.cn/alpha/guide/upgrade)。

### 1.2. Eager execution

TensorFlow 1.X要求用户通过进行`tf.*` API调用，手动将抽象语法树（图形）拼接在一起。然后要求用户通过将一组输出张量和输入张量传递给`session.run()`来手动编译抽象语法树。
TensorFlow 2.0 默认Eager execution模式，马上就执行代码（就像Python通常那样），在2.0中，图形和会话应该像实现细节一样。

Eager execution的一个值得注意的地方是不在需要`tf.control_dependencies()` ，因为所有代码按顺序执行（在`tf.function`中，带有副作用的代码按写入的顺序执行）。

### 1.3. 没有更多的全局变量

TensorFlow 1.X严重依赖于隐式全局命名空间。当你调用`tf.Variable()`时，它会被放入默认图形中，保留在那里，即使你忘记了指向它的Python变量。
然后，您可以恢复该`tf.Variable`，但前提是您知道它已创建的名称，如果您无法控制变量的创建，这很难做到。结果，各种机制激增，试图帮助用户再次找到他们的变量，并寻找框架来查找用户创建的变量：变量范围、全局集合、辅助方法如`tf.get_global_step()`, `tf.global_variables_initializer()`、优化器隐式计算所有可训练变量的梯度等等。

TensorFlow 2.0取消了所有这些机制([Variables 2.0 RFC](https://github.com/tensorflow/community/pull/11))，支持默认机制：跟踪变量！如果你失去了对tf.Variable的追踪，就会垃圾收集回收。

跟踪变量的要求为用户创建了一些额外的工作，但是使用Keras对象（见下文），负担被最小化。

### 1.4. Functions, not sessions

`session.run()`调用几乎就像一个函数调用：指定输入和要调用的函数，然后返回一组输出。
在TensorFlow 2.0中，您可以使用`tf.function()` 来装饰Python函数以将其标记为JIT编译，以便TensorFlow将其作为单个图形运行([Functions 2.0 RFC](https://github.com/tensorflow/community/pull/20))。这种机制允许TensorFlow 2.0获得图形模式的所有好处：

- 性能：可以优化功能（节点修剪，内核融合等）
- 可移植性：该功能可以导出/重新导入([SavedModel 2.0 RFC](https://github.com/tensorflow/community/pull/34))，允许用户重用和共享模块化TensorFlow功能。

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

凭借能够自由穿插Python和TensorFlow代码，我们希望用户能够充分利用Python的表现力。但是可移植的TensorFlow在没有Python解释器的情况下执行-移动端、C++和JS，帮助用户避免在添加 `@tf.function`时重写代码，[AutoGraph](https://tensorflow.google.cn/alpha/guide/autograph)将把Python构造的一个子集转换成它们等效的TensorFlow：

* `for`/`while` -> `tf.while_loop` (支持`break` 和 `continue`)
* `if` -> `tf.cond`
* `for _ in dataset` -> `dataset.reduce`

AutoGraph支持控制流的任意嵌套，这使得高效和简洁地实现许多复杂的ML程序成为可能，比如序列模型、强化学习、自定义训练循环等等。

## 2. 使用TensorFlow 2.0的建议

### 2.1. 将代码重构为更小的函数

TensorFlow 1.X中常见的使用模式是“kitchen sink”策略，在该策略中，所有可能的计算的并集被预先安排好，然后通过`session.run()`对所选的张量进行评估。

TensorFlow 2.0中，用户应该根据需要将代码重构为更小的函数。一般来说，没有必须要使用`tf.function`来修饰这些小函数，只用`tf.function`来修饰高级计算-例如，一个训练步骤，或者模型的前向传递。

### 2.2. 使用Keras层和模型来管理变量

Keras模型和层提供了方便的`variables`和`trainable_variables`属性，它们递归地收集所有的因变量。这使得本地管理变量到使用它们的地方变得非常容易。

对比如下：

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...
  
# 您仍然必须管理w_i和b_i，它们是在代码的其他地方定义的。
```

Keras版本如下：

```python
# 每个图层都可以调用，其签名等价于linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

Keras 层/模型继承自 `tf.train.Checkpointable` 并与`@tf.function`集成，这使得从Keras对象导出保存模型成为可能。
您不必使用Keras的`.fit()` API来利用这些集成。

下面是一个转移学习示例，演示了Keras如何简化收集相关变量子集的工作。假设你正在训练一个拥有共享trunk的multi-headed模型：

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# 训练主要数据集
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    prediction = path1(x)
    loss = loss_fn_head1(prediction, y)
  # 同时优化trunk和head1的权重
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# 微调第二个头部，重用trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    prediction = path2(x)
    loss = loss_fn_head2(prediction, y)
  # 只优化head2的权重，不是trunk的权重
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# 你可以发布trunk计算，以便他人重用。
tf.saved_model.save(trunk, output_path)
```

### 2.3. 结合tf.data.Datesets和@tf.function

当迭代适合内存训练的数据时，可以随意使用常规的Python迭代。除此之外，`tf.data.Datesets`是从磁盘中传输训练数据的最佳方式。
数据集[可迭代（但不是迭代器](https://docs.python.org/3/glossary.html#term-iterable)），就像其他Python迭代器在Eager模式下工作一样。
您可以通过将代码包装在`tf.function()`中来充分利用数据集异步预取/流功能，该代码将Python迭代替换为使用AutoGraph的等效图形操作。

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

如果使用Keras`.fit()`API，就不必担心数据集迭代：

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### 2.4. 利用AutoGraph和Python控制流程

AutoGraph提供了一种将依赖于数据的控制流转换为图形模式等价的方法，如`tf.cond`和`tf.while_loop`。

数据依赖控制流出现的一个常见位置是序列模型。`tf.keras.layers.RNN`封装一个RNN单元格，允许你您静态或动态展开递归。
为了演示，您可以重新实现动态展开如下：

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

有关AutoGraph功能的更详细概述，请参阅[指南](https://tensorflow.google.cn/alpha/guide/autograph).。

### 2.5. 使用tf.metrics聚合数据和tf.summary来记录它

要记录摘要，请使用`tf.summary.(scalar|histogram|...)` 并使用上下文管理器将其重定向到writer。（如果省略上下文管理器，则不会发生任何事情。）与TF 1.x不同，摘要直接发送给writer；没有单独的`merger`操作，也没有单独的`add_summary()`调用，这意味着必须在调用点提供步骤值。

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

要在将数据记录为摘要之前聚合数据，请使用`tf.metrics`，Metrics是有状态的；
当你调用`.result()`时，它们会累计值并返回累计结果。使用`.reset_states()`清除累计值。

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  loss = loss_fn(model(test_x), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

通过将TensorBoard指向摘要日志目录来显示生成的摘要：

```shell
tensorboard --logdir /tmp/summaries
```
