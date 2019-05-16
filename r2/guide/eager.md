
##### Copyright 2018 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Eager Execution


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



TensorFlow's eager execution is an imperative programming environment that
evaluates operations immediately, without building graphs: operations return
concrete values instead of constructing a computational graph to run later. This
makes it easy to get started with TensorFlow and debug models, and it
reduces boilerplate as well. To follow along with this guide, run the code
samples below in an interactive `python` interpreter.

Eager execution is a flexible machine learning platform for research and
experimentation, providing:

* *An intuitive interface*—Structure your code naturally and use Python data
  structures. Quickly iterate on small models and small data.
* *Easier debugging*—Call ops directly to inspect running models and test
  changes. Use standard Python debugging tools for immediate error reporting.
* *Natural control flow*—Use Python control flow instead of graph control
  flow, simplifying the specification of dynamic models.

Eager execution supports most TensorFlow operations and GPU acceleration.

Note: Some models may experience increased overhead with eager execution
enabled. Performance improvements are ongoing, but please
[file a bug](https://github.com/tensorflow/tensorflow/issues) if you find a
problem and share your benchmarks.

## Setup and basic usage

Upgrade to the latest version of TensorFlow:


```
from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf
```

In Tensorflow 2.0, eager execution is enabled by default.


```
tf.executing_eagerly()
```

Now you can run TensorFlow operations and the results will return immediately:


```
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
```

Enabling eager execution changes how TensorFlow operations behave—now they
immediately evaluate and return their values to Python. `tf.Tensor` objects
reference concrete values instead of symbolic handles to nodes in a computational
graph. Since there isn't a computational graph to build and run later in a
session, it's easy to inspect results using `print()` or a debugger. Evaluating,
printing, and checking tensor values does not break the flow for computing
gradients.

Eager execution works nicely with [NumPy](http://www.numpy.org/). NumPy
operations accept `tf.Tensor` arguments. TensorFlow
[math operations](https://www.tensorflow.org/api_guides/python/math_ops) convert
Python objects and NumPy arrays to `tf.Tensor` objects. The
`tf.Tensor.numpy` method returns the object's value as a NumPy `ndarray`.


```
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
```


```
# Broadcasting support
b = tf.add(a, 1)
print(b)
```


```
# Operator overloading is supported
print(a * b)
```


```
# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
```


```
# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

## Dynamic control flow

A major benefit of eager execution is that all the functionality of the host
language is available while your model is executing. So, for example,
it is easy to write [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz):


```
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


```
fizzbuzz(15)
```

This has conditionals that depend on tensor values and it prints these values
at runtime.

## Build a model

Many machine learning models are represented by composing layers. When
using TensorFlow with eager execution you can either write your own layers or
use a layer provided in the `tf.keras.layers` package.

While you can use any Python object to represent a layer,
TensorFlow has `tf.keras.layers.Layer` as a convenient base class. Inherit from
it to implement your own layer, and set `self.dynamic=True` in the constructor if the layer must be executed imperatively:


```
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units
    self.dynamic = True

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)
```

Use `tf.keras.layers.Dense` layer instead  of `MySimpleLayer` above as it has
a superset of its functionality (it can also add a bias).

When composing layers into models you can use `tf.keras.Sequential` to represent
models which are a linear stack of layers. It is easy to use for basic models:


```
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])
```

Alternatively, organize models in classes by inheriting from `tf.keras.Model`.
This is a container for layers that is a layer itself, allowing `tf.keras.Model`
objects to contain other `tf.keras.Model` objects.


```
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

It's not required to set an input shape for the `tf.keras.Model` class since
the parameters are set the first time input is passed to the layer.

`tf.keras.layers` classes create and contain their own model variables that
are tied to the lifetime of their layer objects. To share layer variables, share
their objects.

## Eager training

### Computing gradients

[Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
is useful for implementing machine learning algorithms such as
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training
neural networks. During eager execution, use `tf.GradientTape` to trace
operations for computing gradients later.

`tf.GradientTape` is an opt-in feature to provide maximal performance when
not tracing. Since different operations can occur during each call, all
forward-pass operations get recorded to a "tape". To compute the gradient, play
the tape backwards and then discard. A particular `tf.GradientTape` can only
compute one gradient; subsequent calls throw a runtime error.


```
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```

### Train a model

The following example creates a multi-layer model that classifies the standard
MNIST handwritten digits. It demonstrates the optimizer and layer APIs to build
trainable graphs in an eager execution environment.


```
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)
```


```
# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
```


Even without training, call the model and inspect the output in eager execution:


```
for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())
```

While keras models have a builtin training loop (using the `fit` method), sometimes you need more customization. Here's an example, of a training loop implemented with eager:


```
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
```


```
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


```
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
```

### Variables and optimizers

`tf.Variable` objects store mutable `tf.Tensor` values accessed during
training to make automatic differentiation easier. The parameters of a model can
be encapsulated in classes as variables.

Better encapsulate model parameters by using `tf.Variable` with
`tf.GradientTape`. For example, the automatic differentiation example above
can be rewritten:


```
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
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

## Use objects for state during eager execution

With TF 1.x graph execution, program state (such as the variables) is stored in global
collections and their lifetime is managed by the `tf.Session` object. In
contrast, during eager execution the lifetime of state objects is determined by
the lifetime of their corresponding Python object.

### Variables are objects

During eager execution, variables persist until the last reference to the object
is removed, and is then deleted.


```
if tf.test.is_gpu_available():
  with tf.device("gpu:0"):
    v = tf.Variable(tf.random.normal([1000, 1000]))
    v = None  # v no longer takes up GPU memory
```

### Object-based saving

This section is an abbreviated version of the [guide to training checkpoints](./checkpoints.ipynb).

`tf.train.Checkpoint` can save and restore `tf.Variable`s to and from
checkpoints:


```
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
```


```
x.assign(2.)   # Assign a new value to the variables and save.
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')
```


```
x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0
```

To save and load models, `tf.train.Checkpoint` stores the internal state of objects,
without requiring hidden variables. To record the state of a `model`,
an `optimizer`, and a global step, pass them to a `tf.train.Checkpoint`:


```
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

Note: In many training loops, variables are created after `tf.train.Checkpoint.restore` is called. These variables will be restored as soon as they are created, and assertions are available to ensure that a checkpoint has been fully loaded. See the [guide to training checkpoints](./checkpoints.ipynb) for details.

### Object-oriented metrics

`tf.keras.metrics` are stored as objects. Update a metric by passing the new data to
the callable, and retrieve the result using the `tf.keras.metrics.result` method,
for example:


```
m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```

## Advanced automatic differentiation topics

### Dynamic models

`tf.GradientTape` can also be used in dynamic models. This example for a
[backtracking line search](https://wikipedia.org/wiki/Backtracking_line_search)
algorithm looks like normal NumPy code, except there are gradients and is
differentiable, despite the complex control flow:


```
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

### Custom gradients

Custom gradients are an easy way to override gradients. Within the forward function, define the gradient with respect to the
inputs, outputs, or intermediate results. For example, here's an easy way to clip
the norm of the gradients in the backward pass:


```
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

Custom gradients are commonly used to provide a numerically stable gradient for a
sequence of operations:


```
def log1pexp(x):
  return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

```


```
# The gradient computation works fine at x = 0.
grad_log1pexp(tf.constant(0.)).numpy()
```


```
# However, x = 100 fails because of numerical instability.
grad_log1pexp(tf.constant(100.)).numpy()
```

Here, the `log1pexp` function can be analytically simplified with a custom
gradient. The implementation below reuses the value for `tf.exp(x)` that is
computed during the forward pass—making it more efficient by eliminating
redundant calculations:


```
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


```
# As before, the gradient computation works fine at x = 0.
grad_log1pexp(tf.constant(0.)).numpy()
```


```
# And the gradient computation also works at x = 100.
grad_log1pexp(tf.constant(100.)).numpy()
```

## Performance

Computation is automatically offloaded to GPUs during eager execution. If you
want control over where a computation runs you can enclose it in a
`tf.device('/gpu:0')` block (or the CPU equivalent):


```
import time

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
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

A `tf.Tensor` object can be copied to a different device to execute its
operations:


```
if tf.test.is_gpu_available():
  x = tf.random.normal([10, 10])

  x_gpu0 = x.gpu()
  x_cpu = x.cpu()

  _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
  _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

```

### Benchmarks

For compute-heavy models, such as
[ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)
training on a GPU, eager execution performance is comparable to `tf.function` execution.
But this gap grows larger for models with less computation and there is work to
be done for optimizing hot code paths for models with lots of small operations.

## Work with functions

While eager execution makes development and debugging more interactive,
TensorFlow 1.x style graph execution has advantages for distributed training, performance
optimizations, and production deployment. To bridge this gap, TensorFlow 2.0 introduces `function`s via the `tf.function` API. For more information, see the [Autograph](./autograph.ipynb) guide.
