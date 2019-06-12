---
title: 使用Keras和TensorFlow Hub对电影评论进行文本分类
categories: tensorflow2官方教程
tags: tensorflow2.0教程
top: 1918
abbrlink: tensorflow/tf2-tutorials-keras-basic_text_classification_with_tfhub
---

# 使用Keras和TensorFlow Hub对电影评论进行文本分类 (tensorflow2.0官方教程翻译)

此教程本会将文本形式的影评分为“正面”或“负面”影评。这是一个二元分类（又称为两类分类）的示例，也是一种重要且广泛适用的机器学习问题。

本教程演示了使用TensorFlow Hub和Keras进行迁移学习的基本应用。

数据集使用 [IMDB 数据集](https://tensorflow.google.cn/api_docs/python/tf/keras/datasets/imdb)，其中包含来自互联网电影数据库  https://www.imdb.com/ 的50000 条影评文本。我们将这些影评拆分为训练集（25000 条影评）和测试集（25000 条影评）。训练集和测试集之间达成了平衡，意味着它们包含相同数量的正面和负面影评。

此教程使用[tf.keras](https://www.tensorflow.org/guide/keras)，一种用于在 TensorFlow 中构建和训练模型的高阶 API，以及[TensorFlow Hub](https://www.tensorflow.org/hub)，一个用于迁移学习的库和平台。

有关使用 tf.keras 的更高级文本分类教程，请参阅 [MLCC 文本分类指南](https://developers.google.cn/machine-learning/guides/text-classification/)。

导入库：

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
```

## 1. 下载 IMDB 数据集

[TensorFlow数据集](https://github.com/tensorflow/datasets)上提供了IMDB数据集。以下代码将IMDB数据集下载到您的机器：

```python
# 将训练集分成60％和40％，因此我们最终会得到15,000个训练样本，10,000个验证样本和25,000个测试样本。
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)
```

## 2. 探索数据 

我们花点时间来了解一下数据的格式，每个样本表示电影评论和相应标签的句子，该句子不以任何方式进行预处理。每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。

我们先打印10个样本。

```python
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
```

我们还打印前10个标签。

```python
train_labels_batch
```

## 3. 构建模

神经网络通过堆叠层创建而成，这需要做出三个架构方面的主要决策：

* 如何表示文字？
* 要在模型中使用多少个层？
* 要针对每个层使用多少个隐藏单元？

在此示例中，输入数据由句子组成。要预测的标签是0或1。

表示文本的一种方法是将句子转换为嵌入向量。我们可以使用预先训练的文本嵌入作为第一层，这将具有两个优点：
*  我们不必担心文本预处理，
*  我们可以从迁移学习中受益
*  嵌入具有固定的大小，因此处理起来更简单。

对于此示例，我们将使用来自[TensorFlow Hub](https://www.tensorflow.org/hub) 的预训练文本嵌入模型，名为[google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1).

要达到本教程的目的，还有其他三种预训练模型可供测试：
* [google/tf2-preview/gnews-swivel-20dim-with-oov/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1) 与 [google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)相同，但2.5％的词汇量转换为OOV桶。如果模型的任务和词汇表的词汇不完全重叠，这可以提供帮助。

* [google/tf2-preview/nnlm-en-dim50/1](https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1) 一个更大的模型，具有约1M的词汇量和50个维度。
* [google/tf2-preview/nnlm-en-dim128/1](https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1) 甚至更大的模型，具有约1M的词汇量和128个维度。

让我们首先创建一个使用TensorFlow Hub模型嵌入句子的Keras层，并在几个输入示例上进行尝试。请注意，无论输入文本的长度如何，嵌入的输出形状为：`(num_examples, embedding_dimension)`。

```python
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
```

现在让我们构建完整的模型：

```python
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```

```output
            Model: "sequential" 
            _________________________________________________________________ 
            Layer (type) Output Shape Param # 
            =================================================================
            keras_layer (KerasLayer) (None, 20) 400020 
            _________________________________________________________________ 
            dense (Dense) (None, 16) 336
            _________________________________________________________________ 
            dense_1 (Dense) (None, 1) 17 
            ================================================================= 
            Total params: 400,373 Trainable params: 400,373 Non-trainable params: 0 
            _________________________________________________________________
```

这些图层按顺序堆叠以构建分类器：
1. 第一层是TensorFlow Hub层。该层使用预先训练的保存模型将句子映射到其嵌入向量。我们正在使用的预训练文本嵌入模型([google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1))将句子拆分为标记，嵌入每个标记然后组合嵌入。生成的维度为：`(num_examples, embedding_dimension)`。

2. 这个固定长度的输出矢量通过一个带有16个隐藏单元的完全连接（“密集”）层传输。
3. 最后一层与单个输出节点密集连接。使用`sigmoid`激活函数，该值是0到1之间的浮点数，表示概率或置信度。

让我们编译模型。

### 3.1. 损失函数和优化器

模型在训练时需要一个损失函数和一个优化器。由于这是一个二元分类问题且模型会输出一个概率（应用 S 型激活函数的单个单元层），因此我们将使用 binary_crossentropy 损失函数。

该函数并不是唯一的损失函数，例如，您可以选择 mean_squared_error。但一般来说，binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，在本例中则为实际分布和预测之间的“差距”。

稍后，在探索回归问题（比如预测房价）时，我们将了解如何使用另一个称为均方误差的损失函数。

现在，配置模型以使用优化器和损失函数：

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## 4. 训练模型

用有 512 个样本的小批次训练模型 40 个周期。这将对 x_train 和 y_train 张量中的所有样本进行 40 次迭代。在训练期间，监控模型在验证集的 10000 个样本上的损失和准确率：

```python
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
```

```
...output
            Epoch 20/20
            30/30 [==============================] - 4s 144ms/step - loss: 0.2027 - accuracy: 0.9264 - val_loss: 0.3079 - val_accuracy: 0.8697
```

## 5. 评估模型

我们来看看模型的表现如何。模型会返回两个值：损失（表示误差的数字，越低越好）和准确率。

```python
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
```

```
            loss: 0.324 accuracy: 0.860
```

使用这种相当简单的方法可实现约 87% 的准确率。如果采用更高级的方法，模型的准确率应该会接近 95%。

## 6. 进一步阅读

要了解处理字符串输入的更一般方法，以及更详细地分析训练过程中的准确性和损失，请查看 https://www.tensorflow.org/tutorials/keras/basic_text_classification

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-tutorials-keras-basic_text_classification_with_tfhub.html](https://www.mashangxue123.com/tensorflow/tf2-tutorials-keras-basic_text_classification_with_tfhub.html)
> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/keras/basic_text_classification_with_tfhub](https://tensorflow.google.cn/alpha/tutorials/keras/basic_text_classification_with_tfhub)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/keras/basic_text_classification_with_tfhub.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/keras/basic_text_classification_with_tfhub.md)

