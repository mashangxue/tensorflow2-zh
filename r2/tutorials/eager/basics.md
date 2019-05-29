---
title: tensorflow2.0张量及其操作、numpy兼容、GPU加速 (tensorflow2官方教程翻译）
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 1999
abbrlink: tensorflow/tf2-tutorials-eager-basics
---

# tensorflow2.0张量及其操作、numpy兼容、GPU加速 (tensorflow2官方教程翻译）

> 最新版本：[http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-basics](http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-basics)
> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/eager/basics](https://tensorflow.google.cn/alpha/tutorials/eager/basics)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/basics.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/basics.md)

这是一个基础入门的TensorFlow教程，展示了如何：

* 导入所需的包
* 创建和使用张量
* 使用GPU加速
* 演示 `tf.data.Dataset`

```python
from __future__ import absolute_import, division, print_function
```

## 1. 导入TensorFlow

要开始，请导入tensorflow模块。从TensorFlow 2.0开始，默认情况下用会启用Eager execution，这使得TensorFlow能够实现更加互动的前端，我们将在稍后讨论这些细节。

```python
import tensorflow as tf
```

## 2. 张量

张量是一个多维数组，与NumPy的 `ndarray` 对象类似，`tf.Tensor` 对象具有数据类型和形状，此外，`tf.Tensor` 可以驻留在加速器内存中（如GPU）。TensorFlow提供了丰富的操作库（([tf.add](https://www.tensorflow.org/api_docs/python/tf/add), [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul), [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) 等），它们使用和生成`tf.Tensor`。这些操作会自动转换本机Python类型，例如：

```python
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# 操作符重载也支持
print(tf.square(2) + tf.square(3))
```

```
      tf.Tensor(3, shape=(), dtype=int32) 
      tf.Tensor([4 6], shape=(2,), dtype=int32) 
      tf.Tensor(25, shape=(), dtype=int32) 
      tf.Tensor(6, shape=(), dtype=int32) 
      tf.Tensor(13, shape=(), dtype=int32)
```

每个 `tf.Tensor` 有一个形状和数据类型：

```python
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)
```

```
      tf.Tensor([[2 3]], shape=(1, 2), dtype=int32) (1, 2) <dtype: 'int32'>
```

NumPy数组和 `tf.Tensor` 之间最明显的区别是：

1. 张量可以有加速器内存（如GPU,TPU）支持。

2. 张量是不可改变的。


### 2.1 NumPy兼容性

在TensorFlow的 `tf.Tensor` 和NumPy的 `ndarray` 之间转换很容易：

* TensorFlow操作自动将NumPy ndarray转换为Tensor

* NumPy操作自动将Tensor转换为NumPy ndarray

使用`.numpy（）`方法将张量显式转换为NumPy `ndarrays`。这些转换通常很便宜，因为如果可能的话，数组和`tf.Tensor`共享底层的内存表示。但是，共享底层表示并不总是可行的，因为`tf.Tensor`可以托管在GPU内存中，而NumPy阵列总是由主机内存支持，并且转换涉及从GPU到主机内存的复制。

```python
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())
```

```
    TensorFlow operations convert numpy arrays to Tensors automatically
      tf.Tensor( [[42. 42. 42.] [42. 42. 42.] [42. 42. 42.]], shape=(3, 3), dtype=float64) 
    And NumPy operations convert Tensors to numpy arrays automatically
      [[43. 43. 43.] [43. 43. 43.] [43. 43. 43.]] 
    The .numpy() method explicitly converts a Tensor to a numpy array 
      [[42. 42. 42.] [42. 42. 42.] [42. 42. 42.]]
```

## 3. GPU加速

使用GPU进行计算可以加速许多TensorFlow操作，如果没有任何注释，TensorFlow会自动决定是使用GPU还是CPU进行操作，如果有必要，可以复制CPU和GPU内存之间的张量，操作产生的张量通常由执行操作的设备的存储器支持，例如：

```python
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))
```

### 3.1 设备名称


`Tensor.device`属性提供托管张量内容的设备的完全限定字符串名称。此名称编码许多详细信息，例如正在执行此程序的主机的网络地址的标识符以及该主机中的设备。这是分布式执行TensorFlow程序所必需的。如果张量位于主机上的第N个GPU上，则字符串以 `GPU:<N>`  结尾。
  
### 3.2 显式设备放置

在TensorFlow中，*placement* (放置)指的是如何分配（放置）设备以执行各个操作，如上所述，如果没有提供明确的指导，TensorFlow会自动决定执行操作的设备，并在需要时将张量复制到该设备。但是，可以使用 `tf.device` 上下文管理器将TensorFlow操作显式放置在特定设备上，例如：

```python
import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
```

```
      On CPU: 10 loops: 88.60ms
```

## 4. 数据集

本节使用 [`tf.data.Dataset` API](https://www.tensorflow.org/guide/datasets) 构建管道，以便为模型提供数据。 `tf.data.Dataset`  API用于从简单，可重复使用的部分构建高性能，复杂的输入管道，这些部分将为模型的训练或评估循环提供支持。


### 4.1 创建源数据集

使用其中一个工厂函数（如 [`Dataset.from_tensors`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensors), [`Dataset.from_tensor_slices`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)）或使用从[`TextLineDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset) 或  [`TFRecordDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) 等文件读取的对象创建源数据集。有关详细信息，请参阅[TensorFlow数据集指南](https://www.tensorflow.org/guide/datasets#reading_input_data)。

```python
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
```

### 4.2 应用转换

使用 [`map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map), [`batch`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch), 和 [`shuffle`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)等转换函数将转换应用于数据集记录。

```python
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)
```

### 4.3 迭代（Iterate）

`tf.data.Dataset` 对象支持迭代循环：


```python
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)
```

```
      Elements of ds_tensors:
        tf.Tensor([1 9], shape=(2,), dtype=int32) 
        tf.Tensor([ 4 25], shape=(2,), dtype=int32) 
        tf.Tensor([16 36], shape=(2,), dtype=int32) 
      Elements in ds_file: 
        tf.Tensor([b'Line 1' b'Line 2'], shape=(2,), dtype=string) 
        tf.Tensor([b'Line 3' b' '], shape=(2,), dtype=string)
```
