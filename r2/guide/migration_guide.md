
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

# Convert Your Existing Code to TensorFlow 2.0

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/guide/migration_guide">
    <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
    View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/guide/migration_guide.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/guide/migration_guide.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

It is still possible to run 1.X code, unmodified (except for contrib), in TensorFlow 2.0:

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

However, this does not let you take advantage of many of the improvements made in TensorFlow 2.0. This guide will help you upgrade your code, making it simpler, more performant, and easier to maintain.

## Automatic conversion script

The first step is to try running the [upgrade script](./upgrade.md).

This will do an initial pass at upgrading your code to TensorFlow 2.0. But it can't make your code idiomatic to TensorFlowF 2.0. Your code may still make use of `tf.compat.v1` endpoints to access placeholders, sessions, collections, and other 1.x-style functionality.

## Make the code 2.0-native


This guide will walk through several examples of converting TensorFlow 1.x code to TensorFlow 2.0. These changes will let your code take advantage of performance optimizations and simplified API calls.

In each case, the pattern is:

### 1. Replace `tf.Session.run` calls

Every `tf.Session.run` call should be replaced by a Python function.

* The `feed_dict` and `tf.placeholder`s become function arguments.
* The `fetches` become the function's return value.

You can step-through and debug the function using standard Python tools like `pdb`.

When you're satisfied that it works, add a `tf.function` decorator to make it run efficiently in graph mode. See the [Autograph Guide](autograph.ipynb) for more on how this works.

### 2. Use python objects to track variables and losses

Use `tf.Variable` instead of `tf.get_variable`.

Every `variable_scope` can be converted to a Python object. Typically this will be one of:

* `tf.keras.layers.Layer`
* `tf.keras.Model`
* `tf.Module`

If you need to aggregate lists of variables (like `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`), use the `.variables` and `.trainable_variables` attributes of the `Layer` and `Model` objects.

These `Layer` and `Model` classes implement several other properties that remove the need for global collections. Their `.losses` property can be a replacement for using the `tf.GraphKeys.LOSSES` collection.

See the [keras guides](keras.ipynb) for details.

Warning: Many `tf.compat.v1` symbols  use the global collections implicitly.


### 3. Upgrade your training loops

Use the highest level API that works for your use case.  Prefer `tf.keras.Model.fit` over building your own training loops.

These high level functions manage a lot of the low-level details that might be easy to miss if you write your own training loop. For example, they automatically collect the regularization losses, and set the `training=True` argument when calling the model.

### 4. Upgrade your data input pipelines

Use `tf.data` datasets for data input. Thse objects are efficient, expressive, and integrate well with tensorflow.

They can be passed directly to the `tf.keras.Model.fit` method.

```
model.fit(dataset, epochs=5)
```

They can be iterated over directly standard Python:

```
for example_batch, label_batch in dataset:
    break
```


## Converting models

### Setup


```
from __future__ import absolute_import, division, print_function, unicode_literals
!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf


import tensorflow_datasets as tfds
```

### Low-level variables & operator execution

Examples of low-level API use include:

* using variable scopes to control reuse
* creating variables with `tf.get_variable`.
* accessing collections explicitly
* accessing collections implicitly with methods like :

  * `tf.global_variables`
  * `tf.losses.get_regularization_loss`

* using `tf.placeholder` to set up graph inputs
* executing graphs with `session.run`
* initializing variables manually


#### Before converting

Here is what these patterns may look like in code using TensorFlow 1.x.

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

#### After converting

In the converted code:

* The variables are local Python objects.
* The `forward` function still defines the calculation.
* The `sess.run` call is replaced with a call to `forward`
* The optional `tf.function` decorator can be added for performance.
* The regularizations are calculated manually, without referring to any global collection.
* **No sessions or placeholders.**


```
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)
```


```
out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss = regularizer(W)
```

### Models based on `tf.layers`

The `tf.layers` module is used to contain layer-functions that relied on `tf.variable_scope` to define and reuse variables.

#### Before converting
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

#### After converting

* The simple stack of layers fits neatly into `tf.keras.Sequential`. (For more complex models see [custom layers and models](keras/custom_layers_and_models.ipynb), and [the functional API](keras/functional.ipynb).)
* The model tracks the variables, and regularization losses.
* The conversion was one-to-one because there is a direct mapping from `tf.layers` to `tf.keras.layers`.

Most arguments stayed the same. But notice the differences:

* The `training` argument is passed to each layer by the model when it runs.
* The first argument to the original `model` function (the input `x`) is gone. This is because object layers separate building the model from calling the model.


Also note that:

* If you were using regularizers of initializers from  `tf.contrib`, these have more argument changes than others.
* The code no longer writes to collections, so functions like `tf.losses.get_regularization_loss` will no longer return these values, potentially breaking your training loops.


```
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


```
train_out = model(train_data, training=True)
print(train_out)
```


```
test_out = model(test_data, training=False)
print(test_out)
```


```
# Here are all the trainable variables.
len(model.trainable_variables)
```


```
# Here is the regularization loss.
model.losses
```

### Mixed variables & tf.layers


Existing code often mixes lower-level TF 1.x variables and operations with higher-level `tf.layers`.

#### Before converting
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

#### After converting

To convert this code, follow the pattern of mapping layers to layers as in the previous example.

The `tf.variable_scope` is effectively a layer of its own. So rewrite it as a `tf.keras.layers.Layer`. See [the guide](keras/custom_layers_and_models.ipynb) for details.

The general pattern is:

* Collect layer parameters in `__init__`.
* Build the variables in `build`.
* Execute the calculations in `call`, and return the result.

The `tf.variable_scope` is essentially a layer of its own. So rewrite it as a `tf.keras.layers.Layer`. See [the guide](keras/custom_layers_and_models.ipynb) for details.


```
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

  # Call method will sometimes get used in graph mode,
  # training will get turned into a tensor
  @tf.function
  def call(self, inputs, training=None):
    if training:
      return inputs + self.w
    else:
      return inputs + self.w * 0.5
```


```
custom_layer = CustomLayer()
print(custom_layer([1]).numpy())
print(custom_layer([1], training=True).numpy())
```


```
train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

# Build the model including the custom layer
model = tf.keras.Sequential([
    CustomLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
])

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

```

Some things to note:

* Subclassed Keras models & layers need to run in both v1 graphs (no automatic control dependencies) and in eager mode
  * Wrap the `call()` in a `tf.function()` to get autograph and automatic control dependencies

* Don't forget to accept a `training` argument to `call`.
    * Sometimes it is a `tf.Tensor`
    * Sometimes it is a Python boolean.

* Create model variables in constructor or `def build()` using `self.add_weight()`.
  * In `build` you have access to the input shape, so can create weights with matching shape.
  * Using `tf.keras.layers.Layer.add_weight` allows Keras to track variables and regularization losses.

* Don't keep `tf.Tensors` in your objects.
  * They might get created either in a `tf.function` or in the eager context, and these tensors behave differently.
  * Use `tf.Variable`s for state, they are always usable from both contexts
  * `tf.Tensors` are only for intermediate values.

### A note on Slim & contrib.layers

A large amount of older TensorFlow 1.x code uses the [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) library, which was packaged with TensorFlow 1.x as `tf.contrib.layers`. As a `contrib` module, this is no longer available in TensorFlow 2.0, even in `tf.compat.v1`. Converting code using Slim to TF 2.0 is more involved than converting repositories that use `tf.layers`. In fact, it may make sense to convert your Slim code to `tf.layers` first, then convert to Keras.

* Remove `arg_scopes`, all args need to be explicit
* If you use them, split `normalizer_fn` and `activation_fn` into their own layers
* Separable conv layers map to one or more different Keras layers (depthwise, pointwise, and separable Keras layers)
* Slim and `tf.layers` have different arg names & default values
* Some args have different scales
* If you use Slim pre-trained models, try out `tf.keras.applications` or [TFHub](https://tensorflow.orb/hub)

Some `tf.contrib` layers might not have been moved to core TensorFlow but have instead been moved to the [TF add-ons package](https://github.com/tensorflow/addons).


## Training

There are many ways to feed data to a `tf.keras` model. They will accept Python generators and Numpy arrays as input.

The recomended way to feed data to a model is to use the `tf.data` package, which contains a collection of high performance classes for manipulating data.

If you are still using `tf.queue`, these are only supported as data-structures, not as input pipelines.

### Using Datasets

The [TensorFlow Datasets](https://tensorflow.org/datasets) package (`tfds`) contains utilities for loading predefined datasets as `tf.data.Dataset` objects.

For this example, load the MNISTdataset, using `tfds`:


```
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']
```

Then prepare the data for training:

  * Re-scale each image.
  * Shuffle the order of the examples.
  * Collect batches of images and labels.



```
BUFFER_SIZE = 10 # Use a much larger value for real code.
BATCH_SIZE = 64
NUM_EPOCHS = 5


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label
```

 To keep the example short, trim the dataset to only return 5 batches:


```
train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).take(5)
test_data = mnist_test.map(scale).batch(BATCH_SIZE).take(5)

STEPS_PER_EPOCH = 5

train_data = train_data.take(STEPS_PER_EPOCH)
test_data = test_data.take(STEPS_PER_EPOCH)
```


```
image_batch, label_batch = next(iter(train_data))
```

### Use Keras training loops

If you don't need low level control of your training process, using Keras's built-in `fit`, `evaluate`, and `predict` methods is recomended. These methods provide a uniform interface to train the model regardless of the implementation (sequential,  functional, or sub-classed).

The advantages of these methods include:

* They accept Numpy arrays, Python generators and, `tf.data.Datasets`
* They apply regularization, and activation losses automatically.
* They support `tf.distribute` [for multi-device training](distribute_strategy.ipynb).
* They support arbitrary callables as losses and metrics.
* They support callbacks like `tf.keras.callbacks.TensorBoard`, and custom callbacks.
* They are performant, automatically using TensorFlow graphs.

Here is an example of training a model using a `Dataset`. (For details on how this works see [tutorials](../tutorials).)


```
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

# Model is the full model w/o custom layers
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=NUM_EPOCHS)
loss, acc = model.evaluate(test_data)

print("Loss {}, Accuracy {}".format(loss, acc))
```

### Write your own loop

If the Keras model's training step works for you, but you need more control outside that step, consider using the `tf.keras.model.train_on_batch` method,  in your own data-iteration loop.

Remember: Many things can be implemented as a `tf.keras.Callback`.

This method has many of the advantages of the methods mentioned in the previous section, but gives the user control of the outer loop.

You can also use `tf.keras.model.test_on_batch` or `tf.keras.Model.evaluate` to check performance during training.

Note: `train_on_batch` and `test_on_batch`, by default return the loss and metrics for the single batch. If you pass `reset_metrics=False` they return accumulated metrics and you must remember to appropriately reset the metric accumulators. Also remember that some metrics like `AUC` require `reset_metrics=False` to be calculated correctly.

To continue training the above model:



```
# Model is the full model w/o custom layers
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

### Customize the training step

If you need more flexibility and control, you can have it by implementing your own training loop. There are three steps:

1. Iterate over a Python generator or `tf.data.Dataset` to get batches of examples.
2. Use `tf.GradientTape` to collect gradients.
3. Use a `tf.keras.optimizer` to apply weight updates to the model's variables.

Remember:

* Always include a `training` argument on the `call` method of subclassed layers and models.
* Make sure to call the model with the `training` argument set correctly.
* Depending on usage, model variables may not exist until the model is run on a batch of data.
* You need to manually handle things like regularization losses for the model.

Note the simplifications relative to v1:

* There is no need to run variable initializers. Variables are initialized on creation.
* There is no need to add manual control dependencies. Even in `tf.function` operations act as in eager mode.


```
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

### New-style metrics

In TensorFlow 2.0, metrics are objects. Metric objects work both eagerly and in `tf.function`s. A metric object has the following methods:

* `update_state()` — add new observations
* `result()` —get the current result of the metric, given the observed values
* `reset_states()` — clear all observations.

The object itself is callable. Calling updates the state with new observations, as with `update_state`, and returns the new result of the metric.

You don't have to manually initialize a metric's variables, and because TensorFlow 2.0 has automatic control dependencies, you don't need to worry about those either.

The code below uses a metric to keep track of the mean loss observed within a custom training loop.


```
# Create the metrics
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
  # Update the metrics
  loss_metric.update_state(total_loss)
  accuracy_metric.update_state(labels, predictions)


for epoch in range(NUM_EPOCHS):
  # Reset the metrics
  loss_metric.reset_states()
  accuracy_metric.reset_states()

  for inputs, labels in train_data:
    train_step(inputs, labels)
  # Get the metric results
  mean_loss = loss_metric.result()
  mean_accuracy = accuracy_metric.result()

  print('Epoch: ', epoch)
  print('  loss:     {:.3f}'.format(mean_loss))
  print('  accuracy: {:.3f}'.format(mean_accuracy))

```

## Saving & Loading


### Checkpoint compatibility

TensorFlow 2.0 uses [object-based checkpoints](checkpoints.ipynb).

Old-style name-based checkpoints can still be loaded, if you're careful.
The code conversion process may result in variable name changes, but there are workarounds.

The simplest approach it to line up the names of the new model with the names in the checkpoint:

* Variables still all have a `name` argument you can set.
* Keras models also take a `name` argument as which they set as the prefix for their variables.
* The `tf.name_scope` function can be used to set variable name prefixes. This is very different from `tf.variable_scope`. It only affects names, and  doesn't track variables & reuse.

If that does not work for your use-case, try the `tf.compat.v1.train.init_from_checkpoint` function. It takes an `assignment_map` argument, which specifies the mapping from old names to new names.

Note: Unlike object based checkpoints, which can [defer loading](checkpoints.ipynb#loading_mechanics), name-based checkpoints require that all variables be built when the function is called. Some models defer building variables until you call `build` or run the model on a batch of data.

### Saved models compatibility

There are no significant compatibility concerns for saved models.

* TensorFlow 1.x saved_models work in TensorFlow 2.0.
* TensorFlow 2.0 saved_models even load work in TensorFlow 1.x if all the ops are supported.

## Estimators

### Training with Estimators

Estimators are supported in TensorFlow 2.0.

When you use estimators, you can use `input_fn()`, `tf.estimator.TrainSpec`, and `tf.estimator.EvalSpec` from TensorFlow 1.x.

Here is an example using `input_fn` with train and evaluate specs.

#### Creating the input_fn and train/eval specs


```
# Define the estimator's input_fn
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

# Define train & eval specs
train_spec = tf.estimator.TrainSpec(input_fn=input_fn,
                                    max_steps=STEPS_PER_EPOCH * NUM_EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn,
                                  steps=STEPS_PER_EPOCH)

```

### Using a Keras model definition

There are some differences in how to construct your estimators in TensorFlow 2.0.

We recommend that you define your model using Keras, then use the `tf.keras.model_to_estimator` utility to turn your model into an estimator. The code below shows how to use this utility when creating and training an estimator.


```
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

### Using a custom `model_fn`

If you have an existing custom estimator `model_fn` that you need to maintain, you can convert your `model_fn` to use a Keras model.

However, for compatibility reasons, a custom `model_fn` will still run in 1.x-style graph mode. This means there is no eager execution and no automatic control dependencies.

Using a Keras models in a custom `model_fn` is similar to using it in a custom training loop:

* Set the `training` phase appropriately, based on the `mode` argument.
* Explicitly pass the model's `trainable_variables` to the optimizer.

But there are important differences, relative to a [custom loop](#custom_loop):

* Instead of using `model.losses`, extract the losses using `tf.keras.Model.get_losses_for`.
* Extract the model's updates using `tf.keras.Model.get_updates_for`

Note: "Updates" are changes that need to be applied to a model after each batch. For example, the moving averages of the mean and variance in a `tf.keras.layers.BatchNormalization` layer.

The following code creates an estimator from a custom `model_fn`, illustrating all of these concerns.


```
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

This class was simplified to hold `int`s, instead of `tf.compat.v1.Dimension` objects. So there is no need to call `.value()` to get an `int`.

Individual `tf.compat.v1.Dimension` objects are still accessible from `tf.TensorShape.dims`.



The following demonstrate the differences between TensorFlow 1.x and TensorFlow 2.0.


```
# Create a shape and choose an index
i = 0
shape = tf.TensorShape([16, None, 256])
shape
```

If you had this in TF 1.x:

```python
value = shape[i].value
```

Then do this in TF 2.0:



```
value = shape[i]
value
```

If you had this in TF 1.x:

```python
for dim in shape:
    value = dim.value
    print(value)
```

Then do this in TF 2.0:


```
for value in shape:
  print(value)
```

If you had this in TF 1.x (Or used any other dimension method):

```python
dim = shape[i]
dim.assert_is_compatible_with(other_dim)
```

Then do this in TF 2.0:


```
other_dim = 16
Dimension = tf.compat.v1.Dimension

if shape.rank is None:
  dim = Dimension(None)
else:
  dim = shape.dims[i]
dim.is_compatible_with(other_dim) # or any other dimension method
```


```
shape = tf.TensorShape(None)

if shape:
  dim = shape.dims[i]
  dim.is_compatible_with(other_dim) # or any other dimension method
```

The boolean value of a `tf.TensorShape` is `True` if the rank is known, `False` otherwise.


```
print(bool(tf.TensorShape([])))      # Scalar
print(bool(tf.TensorShape([0])))     # 0-length vector
print(bool(tf.TensorShape([1])))     # 1-length vector
print(bool(tf.TensorShape([None])))  # Unknown-length vector
print(bool(tf.TensorShape([1, 10, 100])))       # 3D tensor
print(bool(tf.TensorShape([None, None, None]))) # 3D tensor with no known dimensions
print()
print(bool(tf.TensorShape(None)))  # A tensor with unknown rank.
```

## Other behavioral changes

There are a few other behavioral changes in TensorFlow 2.0 that you may run into.


### ResourceVariables

TensorFlow 2.0 creates `ResourceVariables` by default, not `RefVariables`.

`ResourceVariables` are locked for writing, and so provide more intuitive consistency guarantees.

* This may change behavior in edge cases.
* This may occasionally create extra copies, can have higher memory usage
* This can be disabled by passing `use_resource=False` to the `tf.Variable` constructor.

### Control Flow

The control flow op implementation has been simplified, and so produces different graphs in TensorFlow 2.0

## Conclusions

The overall process is:

1. Run the upgrade script.
2. Remove contrib symbols.
3. Switch your models to an object oriented style (Keras).
4. Use `tf.keras` or `tf.estimator` training and evaluation loops where you can.
5. Otherwise, use custom loops, but be sure to avoid sessions & collections.


It takes a little work to convert code to idiomatic TensorFlow 2.0, but every change results in:

* Fewer lines of code.
* Increased clarity and simplicity.
* Easier debugging.


