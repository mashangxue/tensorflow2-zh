---
title: 自动微分和梯度带 (tensorflow2.0官方教程翻译）
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 1999
abbrlink: tensorflow/tf2-tutorials-eager-automatic_differentiation
---

# 自动微分和梯度带 (tensorflow2.0官方教程翻译）

在上一个教程中，我们介绍了张量及其操作。在本教程中，我们将介绍自动微分，这是优化机器学习模型的关键技术。


## 1. 导入包

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
```

## 2. 梯度带(Gradient tapes)

TensorFlow提供了 [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) API 用于自动微分(计算与输入变量相关的计算梯度)。
Tensorflow将在 `tf.GradientTape` 上下文中执行的所有操作“records(记录)”到“tape(磁带)”上。然后，TensorFlow使用该磁带和与每个记录操作相关的梯度，使用反向模式微分“记录”计算的梯度。例如：

```python
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
```

您还可以根据在“记录的”tf.GradientTape上下文中计算的中间值请求输出的梯度。

```python
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
```

默认情况下，GradientTape持有的资源会在调用 `GradientTape.gradient()` 方法后立即释放。要在同一计算中计算多个梯度，请创建一个持久梯度带，这允许多次调用 `gradient()` 方法，当磁带对象被垃圾收集时释放资源。例如：

```python
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape
```

### 2.1. 记录控制流程

因为tapes(磁带)在执行时记录操作，所以Python控制流程（例如使用 `if` 和 `while`）自然会被处理：

```python
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

```

### 2.2. 高阶梯度

 `GradientTape` 上下文管理器内的操作将被记录下来，以便自动微分。如果在该上下文中计算梯度，那么梯度计算也会被记录下来。因此，同样的API也适用于高阶梯度。例如:

```python
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
```

## 3. 下一步

在本教程中，我们介绍了TensorFlow中的梯度计算。有了这个，我们就拥有了构建和训练神经网络所需的足够原语。

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-automatic_differentiation.html](https://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-automatic_differentiation.html)

> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/eager/automatic_differentiation](https://tensorflow.google.cn/alpha/tutorials/eager/automatic_differentiation)

> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/automatic_differentiation.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/automatic_differentiation.md)
