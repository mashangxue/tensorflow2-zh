# 高效的TensorFlow 2.0

TensorFlow 2.0中有多处更改，以使TensorFlow用户使用更高效。TensorFlow 2.0删除
[冗余 APIs](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md),
使API更加一致
([统一 RNNs](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md),
[统一优化器](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)),
并通过[Eager execution](https://www.tensorflow.org/guide/eager)模式更好地与Python运行时集成

许多[RFCs](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr)已经解释了TensorFlow 2.0所带来的变化。本指南介绍了TensorFlow 2.0应该是什么样的开发，假设您对TensorFlow 1.x有一定的了解。

## 主要变化的简要总结

### API清理

许多API在tensorflow 2.0中[消失或移动](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)。
一些主要的变化包括删除`tf.app`, `tf.flags`和`tf.logging` ，转而支持现在开源的[absl-py](https://github.com/abseil/abseil-py)，
重新安置`tf.contrib`中的项目，并清理主要的 `tf.*`命名空间，将不常用的函数移动到像 `tf.math`这样的子包中。
一些API已被2.0版本等效替换，如`tf.summary`, `tf.keras.metrics`和`tf.keras.optimizers`。
自动应用这些重命名的最简单方法是使用[v2升级脚本](https://tensorflow.google.cn/alpha/guide/upgrade)。


### Eager execution

TensorFlow 1.X要求用户通过进行`tf.*` API调用，手动将抽象语法树（图形）拼接在一起。然后要求用户通过将一组输出张量和输入张量传递给`session.run()`来手动编译抽象语法树。
TensorFlow 2.0 默认Eager execution模式，马上就执行（就像Python通常那样），在2.0中，图形和会话应该像实现细节一样。

Eager execution的一个值得注意的地方是不在需要`tf.control_dependencies()` ，因为所有代码按顺序执行（在`tf.function`中，带有副作用的代码按写入的顺序执行）。

### 没有更多的全局变量

TensorFlow 1.X严重依赖于隐式全局命名空间。当你调用`tf.Variable()`时，它会被放入默认图形中，保留在那里，即使你忘记了指向它的Python变量。
然后，您可以恢复该`tf.Variable`，但前提是您知道它已创建的名称，如果您无法控制变量的创建，这很难做到。结果，各种机制激增，试图帮助用户再次找到他们的变量，并寻找框架来查找用户创建的变量：变量范围、全局集合、辅助方法如`tf.get_global_step()`, `tf.global_variables_initializer()`、优化器隐式计算所有可训练变量的梯度等等。

TensorFlow 2.0取消了所有这些机制([Variables 2.0 RFC](https://github.com/tensorflow/community/pull/11))，支持默认机制：跟踪变量！如果你失去了对tf.Variable的追踪，就会垃圾收集回收。

跟踪变量的要求为用户创建了一些额外的工作，但是使用Keras对象（见下文），负担被最小化。


### Functions, not sessions

A `session.run()` call is almost like a function call: You specify the inputs
and the function to be called, and you get back a set of outputs. In TensorFlow
2.0, you can decorate a Python function using `tf.function()` to mark it for JIT
compilation so that TensorFlow runs it as a single graph
([Functions 2.0 RFC](https://github.com/tensorflow/community/pull/20)). This
mechanism allows TensorFlow 2.0 to gain all of the benefits of graph mode:

-   Performance: The function can be optimized (node pruning, kernel fusion,
    etc.)
-   Portability: The function can be exported/reimported
    ([SavedModel 2.0 RFC](https://github.com/tensorflow/community/pull/34)),
    allowing users to reuse and share modular TensorFlow functions.

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

With the power to freely intersperse Python and TensorFlow code, we expect that
users will take full advantage of Python's expressiveness. But portable
TensorFlow executes in contexts without a Python interpreter - mobile, C++, and
JS. To help users avoid having to rewrite their code when adding `@tf.function`,
[AutoGraph](autograph.ipynb) will convert a subset of
Python constructs into their TensorFlow equivalents:

*   `for`/`while` -> `tf.while_loop` (`break` and `continue` are supported)
*   `if` -> `tf.cond`
*   `for _ in dataset` -> `dataset.reduce`

AutoGraph supports arbitrary nestings of control flow, which makes it possible
to performantly and concisely implement many complex ML programs such as
sequence models, reinforcement learning, custom training loops, and more.

## Recommendations for idiomatic TensorFlow 2.0

### Refactor your code into smaller functions

A common usage pattern in TensorFlow 1.X was the "kitchen sink" strategy, where
the union of all possible computations was preemptively laid out, and then
selected tensors were evaluated via `session.run()`. In TensorFlow 2.0, users
should refactor their code into smaller functions which are called as needed. In
general, it's not necessary to decorate each of these smaller functions with
`tf.function`; only use `tf.function` to decorate high-level computations - for
example, one step of training, or the forward pass of your model.

### Use Keras layers and models to manage variables

Keras models and layers offer the convenient `variables` and
`trainable_variables` properties, which recursively gather up all dependent
variables. This makes it very easy to manage variables locally to where they are
being used.

Contrast:

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# You still have to manage w_i and b_i, and their shapes are defined far away from the code.
```

with the Keras version:

```python
# Each layer can be called, with a signature equivalent to linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

Keras layers/models inherit from `tf.train.Checkpointable` and are integrated
with `@tf.function`, which makes it possible to directly checkpoint or export
SavedModels from Keras objects. You do not necessarily have to use Keras's
`.fit()` API to take advantage of these integrations.

Here's a transfer learning example that demonstrates how Keras makes it easy to
collect a subset of relevant variables. Let's say you're training a multi-headed
model with a shared trunk:

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# Train on primary dataset
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    prediction = path1(x)
    loss = loss_fn_head1(prediction, y)
  # Simultaneously optimize trunk and head1 weights.
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# Fine-tune second head, reusing the trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    prediction = path2(x)
    loss = loss_fn_head2(prediction, y)
  # Only optimize head2 weights, not trunk weights
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# You can publish just the trunk computation for other people to reuse.
tf.saved_model.save(trunk, output_path)
```

### Combine tf.data.Datasets and @tf.function

When iterating over training data that fits in memory, feel free to use regular
Python iteration. Otherwise, `tf.data.Dataset` is the best way to stream
training data from disk. Datasets are
[iterables (not iterators)](https://docs.python.org/3/glossary.html#term-iterable),
and work just like other Python iterables in Eager mode. You can fully utilize
dataset async prefetching/streaming features by wrapping your code in
`tf.function()`, which replaces Python iteration with the equivalent graph
operations using AutoGraph.

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

If you use the Keras `.fit()` API, you won't have to worry about dataset
iteration.

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### Take advantage of AutoGraph with Python control flow

AutoGraph provides a way to convert data-dependent control flow into graph-mode
equivalents like `tf.cond` and `tf.while_loop`.

One common place where data-dependent control flow appears is in sequence
models. `tf.keras.layers.RNN` wraps an RNN cell, allowing you to either
statically or dynamically unroll the recurrence. For demonstration's sake, you
could reimplement dynamic unroll as follows:

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

For a more detailed overview of AutoGraph's features, see
[the guide](./autograph.ipynb).

### Use tf.metrics to aggregate data and tf.summary to log it

To log summaries, use `tf.summary.(scalar|histogram|...)` and redirect it to a
writer using a context manager. (If you omit the context manager, nothing will
happen.) Unlike TF 1.x, the summaries are emitted directly to the writer; there
is no separate "merge" op and no separate `add_summary()` call, which means that
the `step` value must be provided at the callsite.

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

To aggregate data before logging them as summaries, use `tf.metrics`. Metrics
are stateful; they accumulate values and return a cumulative result when you
call `.result()`. Clear accumulated values with `.reset_states()`.

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

Visualize the generated summaries by pointing TensorBoard at the summary log
directory:


```
tensorboard --logdir /tmp/summaries
```
