
# 探索过拟合和欠拟合

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/keras/overfit_and_underfit"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/overfit_and_underfit.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/overfit_and_underfit.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

在前面的两个例子中（电影影评分类和预测燃油效率），我们看到我们的模型对验证数据的准确性在训练许多周期之后会到达峰值，然后开始下降。

换句话说，我们的模型会过度拟合训练数据，学习如果处理过拟合很重要，尽管通常可以在训练集上实现高精度，但我们真正想要的是开发能够很好泛化测试数据（或之前未见过的数据）的模型。

过拟合的反面是欠拟合，当测试数据仍有改进空间会发生欠拟合，出现这种情况的原因有很多：模型不够强大，过度正则化，或者根本没有经过足够长的时间训练，这意味着网络尚未学习训练数据中的相关模式。

如果训练时间过长，模型将开始过度拟合，并从训练数据中学习模式，而这些模式可能并不适用于测试数据，我们需要取得平衡，了解如何训练适当数量的周期，我们将在下面讨论，这是一项有用的技能。

为了防止过拟合，最好的解决方案是使用更多的训练数据，受过更多数据训练的模型自然会更好的泛化。当没有更多的训练数据时，另外一个最佳解决方案是使用正则化等技术，这些限制了模型可以存储的信息的数据量和类型，如果网络只能记住少量模式，那么优化过程将迫使它专注于最突出的模式，这些模式有更好的泛化性。

在本章节中，我们将探索两种常见的正则化技术：权重正则化和dropout丢弃正则化，并使用它们来改进我们的IMDB电影评论分类。

```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```

## 下载IMDB数据集

我们不会像以前一样使用嵌入，而是对句子进行多重编码。这个模型将很快适应训练集。它将用于证明何时发生过拟合，以及如何处理它。

对我们的列表进行多热编码意味着将它们转换为0和1的向量，具体地说，这将意味着例如将序列[3,5]转换为10000维向量，除了索引3和5的值是1之外，其他全零。

```
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 创建一个全零的形状矩阵 (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # 将results[i]的特定值设为1
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
```

让我们看一下生成的多热矢量，单词索引按频率排序，因此预计索引零附近有更多的1值，我们可以在下图中看到：

```
plt.plot(train_data[0])
```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/overfit_and_underfit_files/output_7_1.png)


## 演示过度拟合

防止过度拟合的最简单方法是减小模型的大小，即模型中可学习参数的数量（由层数和每层单元数决定）。在深度学习中，模型中可学习参数的数量通常被称为模型的“容量”。直观地，具有更多参数的模型将具有更多的“记忆能力”，因此将能够容易地学习训练样本与其目标之间的完美的字典式映射，没有任何泛化能力的映射，但是在对未见过的数据做出预测时这将是无用的。

始终牢记这一点：深度学习模型往往善于适应训练数据，但真正的挑战是泛化，而不是适应。

另一方面，如果网络具有有限的记忆资源，则将不能容易地学习映射。为了最大限度地减少损失，它必须学习具有更强预测能力的压缩表示。同时，如果您使模型太小，则难以适应训练数据。“太多容量”和“容量不足”之间存在平衡。

不幸的是，没有神奇的公式来确定模型的正确大小或架构（就层数而言，或每层的正确大小），您将不得不尝试使用一系列不同的架构。

要找到合适的模型大小，最好从相对较少的层和参数开始，然后开始增加层的大小或添加新层，直到您看到验证损失的收益递减为止。让我们在电影评论分类网络上试试。

我们将仅适用`Dense`层作为基线创建一个简单模型，然后创建更小和更大的版本，并进行比较。

### 创建一个基线模型


```
baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                160016    
_________________________________________________________________
dense_1 (Dense)              (None, 16)                272       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,305
Trainable params: 160,305
Non-trainable params: 0
_________________________________________________________________
```

```
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
```

```
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
25000/25000 - 3s - loss: 0.4664 - accuracy: 0.8135 - binary_crossentropy: 0.4664 - val_loss: 0.3257 - val_accuracy: 0.8808 - val_binary_crossentropy: 0.3257
Epoch 2/20
25000/25000 - 2s - loss: 0.2396 - accuracy: 0.9152 - binary_crossentropy: 0.2396 - val_loss: 0.2851 - val_accuracy: 0.8892 - val_binary_crossentropy: 0.2851
Epoch 3/20
25000/25000 - 2s - loss: 0.1761 - accuracy: 0.9401 - binary_crossentropy: 0.1761 - val_loss: 0.2973 - val_accuracy: 0.8822 - val_binary_crossentropy: 0.2973
Epoch 4/20
25000/25000 - 2s - loss: 0.1434 - accuracy: 0.9514 - binary_crossentropy: 0.1434 - val_loss: 0.3318 - val_accuracy: 0.8720 - val_binary_crossentropy: 0.3318
Epoch 5/20
25000/25000 - 2s - loss: 0.1167 - accuracy: 0.9626 - binary_crossentropy: 0.1167 - val_loss: 0.3434 - val_accuracy: 0.8730 - val_binary_crossentropy: 0.3434
Epoch 6/20
25000/25000 - 2s - loss: 0.0938 - accuracy: 0.9722 - binary_crossentropy: 0.0938 - val_loss: 0.3745 - val_accuracy: 0.8695 - val_binary_crossentropy: 0.3745
Epoch 7/20
25000/25000 - 2s - loss: 0.0750 - accuracy: 0.9788 - binary_crossentropy: 0.0750 - val_loss: 0.4111 - val_accuracy: 0.8652 - val_binary_crossentropy: 0.4111
Epoch 8/20
25000/25000 - 2s - loss: 0.0605 - accuracy: 0.9849 - binary_crossentropy: 0.0605 - val_loss: 0.4511 - val_accuracy: 0.8628 - val_binary_crossentropy: 0.4511
Epoch 9/20
25000/25000 - 2s - loss: 0.0463 - accuracy: 0.9905 - binary_crossentropy: 0.0463 - val_loss: 0.4876 - val_accuracy: 0.8623 - val_binary_crossentropy: 0.4876
Epoch 10/20
25000/25000 - 2s - loss: 0.0367 - accuracy: 0.9937 - binary_crossentropy: 0.0367 - val_loss: 0.5249 - val_accuracy: 0.8594 - val_binary_crossentropy: 0.5249
Epoch 11/20
25000/25000 - 2s - loss: 0.0277 - accuracy: 0.9963 - binary_crossentropy: 0.0277 - val_loss: 0.5637 - val_accuracy: 0.8587 - val_binary_crossentropy: 0.5637
Epoch 12/20
25000/25000 - 2s - loss: 0.0213 - accuracy: 0.9975 - binary_crossentropy: 0.0213 - val_loss: 0.6004 - val_accuracy: 0.8571 - val_binary_crossentropy: 0.6004
Epoch 13/20
25000/25000 - 2s - loss: 0.0163 - accuracy: 0.9989 - binary_crossentropy: 0.0163 - val_loss: 0.6330 - val_accuracy: 0.8561 - val_binary_crossentropy: 0.6330
Epoch 14/20
25000/25000 - 2s - loss: 0.0126 - accuracy: 0.9991 - binary_crossentropy: 0.0126 - val_loss: 0.6648 - val_accuracy: 0.8541 - val_binary_crossentropy: 0.6648
Epoch 15/20
25000/25000 - 2s - loss: 0.0099 - accuracy: 0.9995 - binary_crossentropy: 0.0099 - val_loss: 0.6968 - val_accuracy: 0.8551 - val_binary_crossentropy: 0.6968
Epoch 16/20
25000/25000 - 2s - loss: 0.0079 - accuracy: 0.9997 - binary_crossentropy: 0.0079 - val_loss: 0.7247 - val_accuracy: 0.8540 - val_binary_crossentropy: 0.7247
Epoch 17/20
25000/25000 - 2s - loss: 0.0063 - accuracy: 0.9998 - binary_crossentropy: 0.0063 - val_loss: 0.7531 - val_accuracy: 0.8550 - val_binary_crossentropy: 0.7531
Epoch 18/20
25000/25000 - 2s - loss: 0.0051 - accuracy: 0.9999 - binary_crossentropy: 0.0051 - val_loss: 0.7754 - val_accuracy: 0.8537 - val_binary_crossentropy: 0.7754
Epoch 19/20
25000/25000 - 2s - loss: 0.0043 - accuracy: 0.9999 - binary_crossentropy: 0.0043 - val_loss: 0.8014 - val_accuracy: 0.8540 - val_binary_crossentropy: 0.8014
Epoch 20/20
25000/25000 - 2s - loss: 0.0037 - accuracy: 0.9999 - binary_crossentropy: 0.0037 - val_loss: 0.8219 - val_accuracy: 0.8532 - val_binary_crossentropy: 0.8219
```

### Create a smaller model

Let's create a model with less hidden units to compare against the baseline model that we just created:


```
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()
```

And train the model using the same data:


```
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)
```

### Create a bigger model

As an exercise, you can create an even larger model, and see how quickly it begins overfitting.  Next, let's add to this benchmark a network that has much more capacity, far more than the problem would warrant:


```
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()
```

And, again, train the model using the same data:


```
bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
```

### Plot the training and validation loss

<!--TODO(markdaoust): This should be a one-liner with tensorboard -->

The solid lines show the training loss, and the dashed lines show the validation loss (remember: a lower validation loss indicates a better model). Here, the smaller network begins overfitting later than the baseline model (after 6 epochs rather than 4) and its performance degrades much more slowly once it starts overfitting.


```
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])
```

Notice that the larger network begins overfitting almost right away, after just one epoch, and overfits much more severely. The more capacity the network has, the quicker it will be able to model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

## Strategies to prevent overfitting

### Add weight regularization



You may be familiar with Occam's Razor principle: given two explanations for something, the explanation most likely to be correct is the "simplest" one, the one that makes the least amount of assumptions. This also applies to the models learned by neural networks: given some training data and a network architecture, there are multiple sets of weights values (multiple models) that could explain the data, and simpler models are less likely to overfit than complex ones.

A "simple model" in this context is a model where the distribution of parameter values has less entropy (or a model with fewer parameters altogether, as we saw in the section above). Thus a common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights only to take small values, which makes the distribution of weight values more "regular". This is called "weight regularization", and it is done by adding to the loss function of the network a cost associated with having large weights. This cost comes in two flavors:

* [L1 regularization](https://developers.google.com/machine-learning/glossary/#L1_regularization), where the cost added is proportional to the absolute value of the weights coefficients (i.e. to what is called the "L1 norm" of the weights).

* [L2 regularization](https://developers.google.com/machine-learning/glossary/#L2_regularization), where the cost added is proportional to the square of the value of the weights coefficients (i.e. to what is called the squared "L2 norm" of the weights). L2 regularization is also called weight decay in the context of neural networks. Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization.

L1 regularization introduces sparsity to make some of your weight parameters zero. L2 regularization will penalize the weights parameters without making them sparse—one reason why L2 is more common.

In `tf.keras`, weight regularization is added by passing weight regularizer instances to layers as keyword arguments. Let's add L2 weight regularization now.


```
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)
```

```l2(0.001)``` means that every coefficient in the weight matrix of the layer will add ```0.001 * weight_coefficient_value**2``` to the total loss of the network. Note that because this penalty is only added at training time, the loss for this network will be much higher at training than at test time.

Here's the impact of our L2 regularization penalty:


```
plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])
```

As you can see, the L2 regularized model has become much more resistant to overfitting than the baseline model, even though both models have the same number of parameters.

### Add dropout

Dropout is one of the most effective and most commonly used regularization techniques for neural networks, developed by Hinton and his students at the University of Toronto. Dropout, applied to a layer, consists of randomly "dropping out" (i.e. set to zero) a number of output features of the layer during training. Let's say a given layer would normally have returned a vector [0.2, 0.5, 1.3, 0.8, 1.1] for a given input sample during training; after applying dropout, this vector will have a few zero entries distributed at random, e.g. [0, 0.5,
1.3, 0, 1.1]. The "dropout rate" is the fraction of the features that are being zeroed-out; it is usually set between 0.2 and 0.5. At test time, no units are dropped out, and instead the layer's output values are scaled down by a factor equal to the dropout rate, so as to balance for the fact that more units are active than at training time.

In tf.keras you can introduce dropout in a network via the Dropout layer, which gets applied to the output of layer right before.

Let's add two Dropout layers in our IMDB network to see how well they do at reducing overfitting:


```
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
```


```
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])
```

Adding dropout is a clear improvement over the baseline model.

To recap: here are the most common ways to prevent overfitting in neural networks:

* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add dropout.

And two important approaches not covered in this guide are data-augmentation and batch normalization.
