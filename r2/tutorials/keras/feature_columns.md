---
title: 结构化数据分类
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 1999
abbrlink: tensorflow/tf2-tutorials-keras-feature_columns
---

# 结构化数据分类

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://tensorflow.google.cn/alpha/tutorials/keras/feature_columns">
    <img src="https://tensorflow.google.cn/images/tf_logo_32px.png" />
    View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/feature_columns.ipynb">
    <img src="https://tensorflow.google.cn/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/feature_columns.ipynb">
    <img src="https://tensorflow.google.cn/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

本教程演示了如何对结构化数据进行分类（例如CSV格式的表格数据）。
我们将使用Keras定义模型，并使用[特征列](https://tensorflow.google.cn/guide/feature_columns)作为桥梁，将CSV中的列映射到用于训练模型的特性。
本教程包含完整的代码：

* 使用[Pandas](https://pandas.pydata.org/)加载CSV文件。 .
* 构建一个输入管道，使用[tf.data](https://tensorflow.google.cn/guide/datasets)批处理和洗牌行
* 从CSV中的列映射到用于训练模型的特性。
* 使用Keras构建、训练和评估模型。

## 1. 数据集

我们将使用克利夫兰诊所心脏病基金会提供的一个小[数据集](https://archive.ics.uci.edu/ml/datasets/heart+Disease) 。CSV中有几百行，每行描述一个患者，每列描述一个属性。我们将使用此信息来预测患者是否患有心脏病，该疾病在该数据集中是二元分类任务。

以下是此[数据集的说明](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)。请注意，有数字和分类列。


>Column| Description| Feature Type | Data Type
>------------|--------------------|----------------------|-----------------
>Age | Age in years | Numerical | integer
>Sex | (1 = male; 0 = female) | Categorical | integer
>CP | Chest pain type (0, 1, 2, 3, 4) | Categorical | integer
>Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical | integer
>Chol | Serum cholestoral in mg/dl | Numerical | integer
>FBS | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) | Categorical | integer
>RestECG | Resting electrocardiographic results (0, 1, 2) | Categorical | integer
>Thalach | Maximum heart rate achieved | Numerical | integer
>Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical | integer
>Oldpeak | ST depression induced by exercise relative to rest | Numerical | integer
>Slope | The slope of the peak exercise ST segment | Numerical | float
>CA | Number of major vessels (0-3) colored by flourosopy | Numerical | integer
>Thal | 3 = normal; 6 = fixed defect; 7 = reversable defect | Categorical | string
>Target | Diagnosis of heart disease (1 = true; 0 = false) | Classification | integer

## 2. 导入TensorFlow和其他库

安装sklearn依赖库
```
pip install sklearn
```

```
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
```

## 3. 使用Pandas创建数据帧

[Pandas](https://pandas.pydata.org/) 是一个Python库，包含许多有用的实用程序，用于加载和处理结构化数据。我们将使用Pandas从URL下载数据集，并将其加载到数据帧中。


```
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()
```

## 4. 将数据拆分为训练、验证和测试

我们下载的数据集是一个CSV文件，并将其分为训练，验证和测试集。

```
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
```

```
193 train examples
49 validation examples
61 test examples
```

## 5. 使用tf.data创建输入管道

接下来，我们将使用tf.data包装数据帧，这将使我们能够使用特征列作为桥梁从Pandas数据框中的列映射到用于训练模型的特征。如果我们使用非常大的CSV文件（如此之大以至于它不适合内存），我们将使用tf.data直接从磁盘读取它，本教程不涉及这一点。


```
# 一种从Pandas Dataframe创建tf.data数据集的使用方法 
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
```


```
batch_size = 5 # 小批量用于演示目的
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## 6. 理解输入管道

现在我们已经创建了输入管道，让我们调用它来查看它返回的数据的格式，我们使用了一小批量来保持输出的可读性。


```
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )
```

```
Every feature: ['age', 'chol', 'fbs', 'ca', 'slope', 'restecg', 'sex', 'thal', 'thalach', 'oldpeak', 'exang', 'cp', 'trestbps']
A batch of ages: tf.Tensor([58 52 56 35 59], shape=(5,), dtype=int32)
A batch of targets: tf.Tensor([1 0 1 0 0], shape=(5,), dtype=int32)
```

我们可以看到数据集返回一个列名称（来自数据帧），该列表映射到数据帧中行的列值。

## 7. 演示几种类型的特征列

TensorFlow提供了许多类型的特性列。在本节中，我们将创建几种类型的特性列，并演示它们如何从dataframe转换列。

```
# 我们将使用此批处理来演示几种类型的特征列 
example_batch = next(iter(train_ds))[0]

# 用于创建特征列和转换批量数据 
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())
```

### 7.1. 数字列

特征列的输出成为模型的输入（使用上面定义的演示函数，我们将能够准确地看到数据帧中每列的转换方式），[数字列](https://tensorflow.google.cn/api_docs/python/tf/feature_column/numeric_column)是最简单的列类型，它用于表示真正有价值的特征，使用此列时，模型将从数据帧中接收未更改的列值。

```
age = feature_column.numeric_column("age")
demo(age)
```

```
[[58.]
 [52.]
 [56.]
 [35.]
 [59.]]
 ```

在心脏病数据集中，数据帧中的大多数列都是数字。

### Bucketized列（桶列）

通常，您不希望将数字直接输入模型，而是根据数值范围将其值分成不同的类别，考虑代表一个人年龄的原始数据，我们可以使用[bucketized列](https://tensorflow.google.cn/api_docs/python/tf/feature_column/bucketized_column)将年龄分成几个桶，而不是将年龄表示为数字列。
请注意，下面的one-hot(独热编码)值描述了每行匹配的年龄范围。

```
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)
```

```
[[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
 ```

### 7.2. 分类列

在该数据集中，thal表示为字符串（例如“固定”，“正常”或“可逆”），我们无法直接将字符串提供给模型，相反，我们必须首先将它们映射到数值。分类词汇表列提供了一种将字符串表示为独热矢量的方法（就像上面用年龄段看到的那样）。词汇表可以使用[categorical_column_with_vocabulary_list](https://tensorflow.google.cn/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list)作为列表传递，或者使用[categorical_column_with_vocabulary_file](https://tensorflow.google.cn/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file)从文件加载。


```
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)
```

```
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
 ```
 
在更复杂的数据集中，许多列将是分类的（例如字符串），在处理分类数据时，特征列最有价值。虽然此数据集中只有一个分类列，但我们将使用它来演示在处理其他数据集时可以使用的几种重要类型的特征列。

### 嵌入列

假设我们不是只有几个可能的字符串，而是每个类别有数千（或更多）值。由于多种原因，随着类别数量的增加，使用独热编码训练神经网络变得不可行，我们可以使用嵌入列来克服此限制。
[嵌入列](https://tensorflow.google.cn/api_docs/python/tf/feature_column/embedding_column)不是将数据表示为多维度的独热矢量，而是将数据表示为低维密集向量，其中每个单元格可以包含任意数字，而不仅仅是0或1.嵌入的大小（在下面的例子中是8）是必须调整的参数。

关键点：当分类列具有许多可能的值时，最好使用嵌入列，我们在这里使用一个用于演示目的，因此您有一个完整的示例，您可以在将来修改其他数据集。


```
# 请注意，嵌入列的输入是我们先前创建的分类列 
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)
```

```
[[-0.01019966  0.23583987  0.04172783  0.34261808 -0.02596842  0.05985594
   0.32729048 -0.07209085]
 [ 0.08829682  0.3921798   0.32400072  0.00508362 -0.15642034 -0.17451124
   0.12631968  0.15029909]
 [-0.01019966  0.23583987  0.04172783  0.34261808 -0.02596842  0.05985594
   0.32729048 -0.07209085]
 [-0.01019966  0.23583987  0.04172783  0.34261808 -0.02596842  0.05985594
   0.32729048 -0.07209085]
 [-0.01019966  0.23583987  0.04172783  0.34261808 -0.02596842  0.05985594
   0.32729048 -0.07209085]]
```

### 哈希特征列

Another way to represent a categorical column with a large number of values is to use a [categorical_column_with_hash_bucket](https://tensorflow.google.cn/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket). This feature column calculates a hash value of the input, then selects one of the `hash_bucket_size` buckets to encode a string. When using this column, you do not need to provide the vocabulary, and you can choose to make the number of hash_buckets significantly smaller than the number of actual categories to save space.

表示具有大量值的分类列的另一种方法是使用[categorical_column_with_hash_bucket](https://tensorflow.google.cn/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket).
此特征列计算输入的哈希值，然后选择一个`hash_bucket_size`存储桶来编码字符串，使用此列时，您不需要提供词汇表，并且可以选择使`hash_buckets`的数量远远小于实际类别的数量以节省空间。

关键点：该技术的一个重要缺点是可能存在冲突，其中不同的字符串被映射到同一个桶，实际上，无论如何，这对某些数据集都有效。


```
thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))
```

```
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
```

### 交叉特征列

将特征组合成单个特征（也称为[特征交叉](https://developers.google.com/machine-learning/glossary/#feature_cross)），使模型能够为每个特征组合学习单独的权重。
在这里，我们将创建一个age和thal交叉的新功能，
请注意，`crossed_column`不会构建所有可能组合的完整表（可能非常大），相反，它由`hashed_column`支持，因此您可以选择表的大小。


```
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))
```

```
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
```


## 选择要使用的列

我们已经了解了如何使用几种类型的特征列，现在我们将使用它们来训练模型。本教程的目标是向您展示使用特征列所需的完整代码（例如，机制），我们选择了几列来任意训练我们的模型。

关键点：如果您的目标是建立一个准确的模型，请尝试使用您自己的更大数据集，并仔细考虑哪些特征最有意义，以及如何表示它们。

```
feature_columns = []

# numeric 数字列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator 指示符列 
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding 嵌入列 
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed 交叉列 
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
```

### 创建特征层

现在我们已经定义了我们的特征列，我们将使用[DenseFeatures](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/layers/DenseFeatures)层将它们输入到我们的Keras模型中。


```
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

之前，我们使用小批量大小来演示特征列的工作原理，我们创建了一个具有更大批量的新输入管道。


```
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## 创建、编译和训练模型


```
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)
```

训练过程的输出
```
Epoch 1/5
7/7 [==============================] - 1s 79ms/step - loss: 3.8492 - accuracy: 0.4219 - val_loss: 2.7367 - val_accuracy: 0.7143
......
Epoch 5/5
7/7 [==============================] - 0s 34ms/step - loss: 0.6200 - accuracy: 0.7377 - val_loss: 0.6288 - val_accuracy: 0.6327

<tensorflow.python.keras.callbacks.History at 0x7f48c044c5f8>
```

测试
```
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
```

```
2/2 [==============================] - 0s 19ms/step - loss: 0.5538 - accuracy: 0.6721
Accuracy 0.6721311
```

关键点：通常使用更大更复杂的数据集进行深度学习，您将看到最佳结果。使用像这样的小数据集时，我们建议使用决策树或随机森林作为强基线。

本教程的目标不是为了训练一个准确的模型，而是为了演示使用结构化数据的机制，因此您在将来使用自己的数据集时需要使用代码作为起点。

## 下一步

了解有关分类结构化数据的更多信息的最佳方法是亲自尝试，我们建议找到另一个可以使用的数据集，并训练模型使用类似于上面的代码对其进行分类，要提高准确性，请仔细考虑模型中包含哪些特征以及如何表示这些特征。

