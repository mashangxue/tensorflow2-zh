---
title: tf.function和AutoGraph (tensorflow2.0官方教程翻译）
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 1999
abbrlink: tensorflow/tf2-tutorials-eager-tf_function
---

# tf.function和 AutoGraph (tensorflow2.0官方教程翻译）

> 最新版本：[http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-tf_function](http://www.mashangxue123.com/tensorflow/tf2-tutorials-eager-tf_function)

> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/eager/tf_function](https://tensorflow.google.cn/alpha/tutorials/eager/tf_function)

> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/tf_function.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/eager/tf_function.md)

在TensorFlow 2.0中，默认情况下会打开eager execution，这为您提供了一个非常直观和灵活的用户界面（运行一次性操作更容易，更快）但这可能会牺牲性能和可部署性。

为了获得最佳性能并使您的模型可以在任何地方部署，我们提供了 `tf.function` 作为您可以用来从程序中生成图的工具。多亏了AutoGraph，大量的Python代码可以与tf.function一起工作。但仍有一些陷阱需要警惕。

主要的要点和建议是：

- 不要依赖Python副作用，如对象变异或列表追加。

- tf.function最适合TF操作，而不是NumPy操作或Python原语。

- 如果有疑问，`for x in y` 习语可能会有效。

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}: {}'.format(error_class, e))
  except Exception as e:
    print('Got unexpected exception \n  {}: {}'.format(type(e), e))
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
```

你定义的 `tf.function` 就像一个核心TensorFlow操作：你可以急切地执行它，你可以在图中使用它，它有梯度等。

```python
# A function is like an op

@tf.function
def add(a, b):
  return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
```

```
      <tf.Tensor: id=14, shape=(2, 2), dtype=float32, numpy= array([[2., 2.], [2., 2.]], dtype=float32)>
```


```python
# Functions have gradients

@tf.function
def add(a, b):
  return a + b

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add(v, 1.0)
tape.gradient(result, v)
```

```
      <tf.Tensor: id=40, shape=(), dtype=float32, numpy=1.0>
```


```python
# You can use functions inside functions

@tf.function
def dense_layer(x, w, b):
  return add(tf.matmul(x, w), b)

dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
```

```
  <tf.Tensor: id=67, shape=(3, 2), dtype=float32, numpy= array([[3., 3.], [3., 3.], [3., 3.]], dtype=float32)>
```


## 追踪和多态性

Python的动态类型意味着您可以使用各种参数类型调用函数，Python将在每个场景中执行不同的操作。
另一方面，TensorFlow图需要静态dtypes和形状尺寸。`tf.function` 通过在必要时回溯函数生成正确的图来弥补这一差距。`tf.function` 使用的大多数微妙之处源于这种回溯行为。

您可以使用不同类型的参数调用函数来查看正在发生的事情。

```python
# Functions are polymorphic

@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

```

```
      Tracing with Tensor("a:0", shape=(), dtype=int32) tf.Tensor(2, shape=(), dtype=int32) 
      Tracing with Tensor("a:0", shape=(), dtype=float32) tf.Tensor(2.2, shape=(), dtype=float32) 
      Tracing with Tensor("a:0", shape=(), dtype=string) tf.Tensor(b'aa', shape=(), dtype=string)
```

要控制跟踪行为，请使用以下技术：

- 创建一个新的`tf.function`：保证单独的`tf.function`对象不共享跟踪。

- 使用`get_concrete_function`方法获取特定的跟踪 

- 调用`tf.function`时指定`input_signature`以确保只构建一个函数图


```python
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
  double_strings(tf.constant(1))
```

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(tf.equal(x % 2, 0), x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([[1, 2], [3, 4]]))

```

## 什么时候回溯？

多态 `tf.function` 保持跟踪生成的具体函数的缓存。缓存键实际上是从函数args和kwargs生成的键的元组。为`tf.Tensor`参数生成的关键是它的形状和类型。为Python原语生成的密钥是它的值。对于所有其他Python类型，键基于对象`id（）`，以便为每个类的实例独立跟踪方法。将来，TensorFlow可以为Python对象添加更复杂的缓存，可以安全地转换为张量。

## Python还是Tensor args？

通常，Python参数用于控制超参数和图构造。例如，`num_layers = 10 `或 `training = True` 或`nonlinearity ='relu'`。因此，如果Python参数发生变化，那么您必须回溯图形是有道理的。

但是，Python参数可能不会用于控制图构造。在这些情况下，Python值的变化可能会触发不必要的回溯。举例来说，这个训练循环，AutoGraph将动态展开。尽管存在多个跟踪，但生成的图实际上是相同的，因此这有点低效。

```python
def train_one_step():
  pass

@tf.function
def train(num_steps):
  print("Tracing with num_steps = {}".format(num_steps))
  for _ in tf.range(num_steps):
    train_one_step()

train(num_steps=10)
train(num_steps=20)

```
      Tracing with num_steps = 10 Tracing with num_steps = 20
```

```

如果它们不影响生成的图的形状，简单的解决方法是将参数转换为Tensors。

```python
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```

```
      Tracing with num_steps = Tensor("num_steps:0", shape=(), dtype=int32)
```

## `tf.function`中的附作用

> “副作用” 指“在满足主要功能（主作用？）的同时，顺便完成了一些其他的副要功能”，也可翻译为“附作用”

通常，Python附作用（如打印或变异对象）仅在跟踪期间发生。那你如何可靠地触发`tf.function`的附作用呢？

一般的经验法则是仅使用Python副作用来调试跟踪。另外，TensorFlow操作如`tf.Variable.assign`，`tf.print`和`tf.summary`是确保TensorFlow运行时，在每次调用时，跟踪和执行代码的最佳方法。通常使用函数样式将产生最佳效果。

```python
@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)

f(1)
f(1)
f(2)

```

```
  Traced with 1 Executed with 1 Executed with 1 
  Traced with 2 Executed with 2
```

如果你想在每次调用 `tf.function` 期间执行Python代码，`tf.py_function`就是一个退出舱口。`tf.py_function`的缺点是它不可移植或特别高效，也不能在分布式（多GPU，TPU）设置中很好地工作。此外，由于必须将`tf.py_function`连接到图中，它会将所有输入/输出转换为张量。

```python
external_list = []

def side_effect(x):
  print('Python side effect')
  external_list.append(x)

@tf.function
def f(x):
  tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)
assert len(external_list) == 3
# .numpy() call required because py_function casts 1 to tf.constant(1)
assert external_list[0].numpy() == 1

```

## 谨防Python状态

许多Python功能（如生成器和迭代器）依赖于Python运行时来跟踪状态。通常，虽然这些构造在Eager模式下按预期工作，但由于跟踪行为，在`tf.function`中会发生许多意外情况。

举一个例子，推进迭代器状态是一个Python副作用，因此只在跟踪期间发生。

```python
external_var = tf.Variable(0)
@tf.function
def buggy_consume_next(iterator):
  external_var.assign_add(next(iterator))
  tf.print("Value of external_var:", external_var)

iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)

```

如果在tf.function中生成并完全使用了迭代器，那么它应该可以正常工作。但是，整个迭代器可能正在被跟踪，这可能导致一个巨大的图。这可能就是你想要的。但是如果你正在训练一个表示为Python列表的大型内存数据集，那么这可以生成一个非常大的图，并且`tf.function`不太可能产生加速。

如果你想迭代Python数据，最安全的方法是将它包装在tf.data.Dataset中并使用`for x in y`惯用法。当`y`是张量或tf.data.Dataset时，AutoGraph特别支持安全地转换`for`循环。

```python
def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
  loss = tf.constant(0)
  for x, y in dataset:
    loss += tf.abs(y - x) # Some dummy computation.
  return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
```


在数据集中包装Python / Numpy数据时，请注意 `tf.data.Dataset.from_generator` 与 `tf.data.Dataset.from_tensors`。前者将数据保存在Python中并通过 `tf.py_function` 获取，这可能会影响性能，而后者会将数据的副本捆绑为图中的一个大的 `tf.constant()` 节点，这可以有记忆含义。

通过 TFRecordDataset/CsvDataset等从文件中读取数据，是最有效的数据消费方式，因为TensorFlow本身可以管理数据的异步加载和预取，而不必涉及Python。

## 自动控制依赖项

在一般数据流图上，作为编程模型的函数，一个非常吸引人的特性是函数可以为运行时提供有关代码预期行为的更多信息。

例如，当编写具有多个读取和写入相同变量的代码时，数据流图可能不会自然地编码最初预期的操作顺序。在`tf.function`中，我们通过引用原始Python代码中语句的执行顺序来解决执行顺序中的歧义。这样，`tf.function` 中的有状态操作的排序复制了Eager模式的语义。

这意味着不需要添加手动控制依赖项;`tf.function`非常智能，可以为代码添加最小的必要和足够的控制依赖关系，以便正确运行。

```python
# Automatic control dependencies

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
  a.assign(y * b)
  b.assign_add(x * a)
  return a + b

f(1.0, 2.0)  # 10.0

```

```
      <tf.Tensor: id=466, shape=(), dtype=float32, numpy=10.0>
```

## 变量

我们可以使用相同的想法来利用代码的预期执行顺序，以便在`tf.function`中非常容易地创建和使用变量。但是有一个非常重要的警告，即使用变量，可以编写在急切模式和图形模式下表现不同的代码。

具体来说，每次调用创建一个新变量时都会发生这种情况。由于跟踪语义，`tf.function`将在每次调用时重用相同的变量，但是eager模式将在每次调用时创建一个新变量。为了防止这个错误，`tf.function`会在检测到危险变量创建行为时引发错误。

```python
@tf.function
def f(x):
  v = tf.Variable(1.0)
  v.assign_add(x)
  return v

with assert_raises(ValueError):
  f(1.0)
```

```
      Caught expected exception <class 'ValueError'>: tf.function-decorated function tried to create variables on non-first call.
```

```python
# Non-ambiguous code is ok though

v = tf.Variable(1.0)

@tf.function
def f(x):
  return v.assign_add(x)

print(f(1.0))  # 2.0
print(f(2.0))  # 4.0

```

```
      tf.Tensor(2.0, shape=(), dtype=float32) 
      tf.Tensor(4.0, shape=(), dtype=float32)
```


```python
# You can also create variables inside a tf.function as long as we can prove
# that those variables are created only the first time the function is executed.

class C: pass
obj = C(); obj.v = None

@tf.function
def g(x):
  if obj.v is None:
    obj.v = tf.Variable(1.0)
  return obj.v.assign_add(x)

print(g(1.0))  # 2.0
print(g(2.0))  # 4.0
```

```
      tf.Tensor(2.0, shape=(), dtype=float32) 
      tf.Tensor(4.0, shape=(), dtype=float32)
```

```python
# Variable initializers can depend on function arguments and on values of other
# variables. We can figure out the right initialization order using the same
# method we use to generate control dependencies.

state = []
@tf.function
def fn(x):
  if not state:
    state.append(tf.Variable(2.0 * x))
    state.append(tf.Variable(state[0] * 3.0))
  return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))

```

```
      tf.Tensor(12.0, shape=(), dtype=float32) 
      tf.Tensor(36.0, shape=(), dtype=float32)
```

# Using AutoGraph

[autograph](https://www.tensorflow.org/guide/autograph) 库与`tf.function`完全集成，它将重写依赖于Tensors的条件和循环，以便在图中动态运行。

`tf.cond`和`tf.while_loop`继续使用`tf.function`，但是当以命令式方式编写时，具有控制流的代码通常更容易编写和理解。


```python
# Simple loop

@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

f(tf.random.uniform([5]))
```


```python
# If you're curious you can inspect the code autograph generates.
# It feels like reading assembly language, though.

def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

print(tf.autograph.to_code(f))
```

## AutoGraph：条件

AutoGraph会将`if`语句转换为等效的`tf.cond`调用。
如果条件是Tensor，则进行此替换。否则，在跟踪期间执行条件。

```python
def test_tf_cond(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'cond' for node in g.as_graph_def().node):
    print("{}({}) uses tf.cond.".format(
        f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) executes normally.".format(
        f.__name__, ', '.join(map(str, args))))

```


```python
@tf.function
def hyperparam_cond(x, training=True):
  if training:
    x = tf.nn.dropout(x, rate=0.5)
  return x

@tf.function
def maybe_tensor_cond(x):
  if x < 0:
    x = -x
  return x

test_tf_cond(hyperparam_cond, tf.ones([1], dtype=tf.float32))
test_tf_cond(maybe_tensor_cond, tf.constant(-1))
test_tf_cond(maybe_tensor_cond, -1)

```

`tf.cond`有许多微妙之处。

- 它的工作原理是跟踪条件的两边，然后根据条件在运行时选择适当的分支。跟踪双方可能导致意外执行Python代码

- 它要求如果一个分支创建下游使用的张量，另一个分支也必须创建该张量。

```python
@tf.function
def f():
  x = tf.constant(0)
  if tf.constant(True):
    x = x + 1
    print("Tracing `then` branch")
  else:
    x = x - 1
    print("Tracing `else` branch")
  return x

f()
```


```python
@tf.function
def f():
  if tf.constant(True):
    x = tf.ones([3, 3])
  return x

# Throws an error because both branches need to define `x`.
with assert_raises(ValueError):
  f()
```

## AutoGraph和循环

AutoGraph有一些简单的转换循环规则。

- `for`: 如果iterable是张量，则转换

- `while`: 如果while条件取决于张量，则转换

如果转换了循环，它将使用`tf.while_loop`动态展开，或者在 `for x in tf.data.Dataset` 的特殊情况下，转换为 `tf.data.Dataset.reduce`。

如果未转换循环，则将静态展开。

```python
def test_dynamically_unrolled(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'while' for node in g.as_graph_def().node):
    print("{}({}) uses tf.while_loop.".format(
        f.__name__, ', '.join(map(str, args))))
  elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
    print("{}({}) uses tf.data.Dataset.reduce.".format(
        f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) gets unrolled.".format(
        f.__name__, ', '.join(map(str, args))))

```


```python
@tf.function
def for_in_range():
  x = 0
  for i in range(5):
    x += i
  return x

@tf.function
def for_in_tfrange():
  x = tf.constant(0, dtype=tf.int32)
  for i in tf.range(5):
    x += i
  return x

@tf.function
def for_in_tfdataset():
  x = tf.constant(0, dtype=tf.int64)
  for i in tf.data.Dataset.range(5):
    x += i
  return x

test_dynamically_unrolled(for_in_range)
test_dynamically_unrolled(for_in_tfrange)
test_dynamically_unrolled(for_in_tfdataset)

```

```
      for_in_range() gets unrolled. 
      for_in_tfrange() uses tf.while_loop. 
      for_in_tfdataset() uses tf.data.Dataset.reduce.
```

```python
@tf.function
def while_py_cond():
  x = 5
  while x > 0:
    x -= 1
  return x

@tf.function
def while_tf_cond():
  x = tf.constant(5)
  while x > 0:
    x -= 1
  return x

test_dynamically_unrolled(while_py_cond)
test_dynamically_unrolled(while_tf_cond)
```

```
      while_py_cond() gets unrolled. 
      while_tf_cond() uses tf.while_loop.
```

如果你有一个取决于张量的`break`或早期`return`子句，那么顶级条件或者iterable也应该是一个张量。

```python
@tf.function
def buggy_while_py_true_tf_break(x):
  while True:
    if tf.equal(x, 0):
      break
    x -= 1
  return x

@tf.function
def while_tf_true_tf_break(x):
  while tf.constant(True):
    if tf.equal(x, 0):
      break
    x -= 1
  return x

with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)
test_dynamically_unrolled(while_tf_true_tf_break, 5)

@tf.function
def buggy_py_for_tf_break():
  x = 0
  for i in range(5):
    if tf.equal(i, 3):
      break
    x += i
  return x

@tf.function
def tf_for_tf_break():
  x = 0
  for i in tf.range(5):
    if tf.equal(i, 3):
      break
    x += i
  return x

with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_py_for_tf_break)
test_dynamically_unrolled(tf_for_tf_break)



```

为了累积动态展开循环的结果，你需要使用`tf.TensorArray`。

```python
batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
  return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
  # [batch, time, features] -> [time, batch, features]
  input_data = tf.transpose(input_data, [1, 0, 2])
  max_seq_len = input_data.shape[0]

  states = tf.TensorArray(tf.float32, size=max_seq_len)
  state = initial_state
  for i in tf.range(max_seq_len):
    state = rnn_step(input_data[i], state)
    states = states.write(i, state)
  return tf.transpose(states.stack(), [1, 0, 2])
  
dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))
```

与`tf.cond`一样，`tf.while_loop`也带有许多细微之处。

- 由于循环可以执行0次，因此必须在循环上方初始化在while_loop下游使用的所有张量 

- 所有循环变量的shape/dtypes必须与每次迭代保持一致

```python
@tf.function
def buggy_loop_var_uninitialized():
  for i in tf.range(3):
    x = i
  return x

@tf.function
def f():
  x = tf.constant(0)
  for i in tf.range(3):
    x = i
  return x

with assert_raises(ValueError):
  buggy_loop_var_uninitialized()
f()
```


```python
@tf.function
def buggy_loop_type_changes():
  x = tf.constant(0, dtype=tf.float32)
  for i in tf.range(3): # Yields tensors of type tf.int32...
    x = i
  return x

with assert_raises(tf.errors.InvalidArgumentError):
  buggy_loop_type_changes()
```


```python
@tf.function
def buggy_concat():
  x = tf.ones([0, 10])
  for i in tf.range(5):
    x = tf.concat([x, tf.ones([1, 10])], axis=0)
  return x

with assert_raises(ValueError):
  buggy_concat()
  
@tf.function
def concat_with_padding():
  x = tf.zeros([5, 10])
  for i in tf.range(5):
    x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4-i, 10])], axis=0)
    x.set_shape([5, 10])
  return x

concat_with_padding()

```

## 下一步

现在重新访问早期的教程并尝试使用 `tf.function` 加速代码！
