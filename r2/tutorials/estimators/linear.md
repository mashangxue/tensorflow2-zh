---
title: 使用 Estimator 构建线性模型
tags: 
    - tensorflow2.0
categories: 
    - tensorflow2.0官方文档
date: 2019-05-10
abbrlink: tensorflow/tensorflow2-tutorials-estimators-linear
---

# 使用 Estimator 构建线性模型

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/estimators/linear"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/estimators/linear.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/estimators/linear.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

## 1. 概述

这个端到端的演练使用`tf.estimator` API训练逻辑回归模型。该模型通常用作其他更复杂算法的基准。
Estimator 是可扩展性最强且面向生产的 TensorFlow 模型类型。如需了解详情，请参阅 [Estimator 指南](https://www.tensorflow.org/guide/estimators)。

## 2. 安装和导入

安装sklearn命令:  `pip install sklearn`

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
```

## 3. 加载泰坦尼克号数据集

您将使用泰坦尼克数据集，其以预测乘客的生存(相当病态)为目标，给出性别、年龄、阶级等特征。

```python
import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# 加载数据集
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
```

## 4. 探索数据

数据集包含以下特征：

```python
dftrain.head()
```

|   | sex    | age  | n_siblings_spouses | parch | fare    | class | deck    | embark_town | alone |
|---|--------|------|--------------------|-------|---------|-------|---------|-------------|-------|
| 0 | male   | 22.0 | 1                  | 0     | 7.2500  | Third | unknown | Southampton | n     |
| 1 | female | 38.0 | 1                  | 0     | 71.2833 | First | C       | Cherbourg   | n     |
| 2 | female | 26.0 | 0                  | 0     | 7.9250  | Third | unknown | Southampton | y     |
| 3 | female | 35.0 | 1                  | 0     | 53.1000 | First | C       | Southampton | n     |
| 4 | male   | 28.0 | 0                  | 0     | 8.4583  | Third | unknown | Queenstown  | y     |


```python
dftrain.describe()
```

|       | age        | n_siblings_spouses | parch      | fare       |
|-------|------------|--------------------|------------|------------|
| count | 627.000000 | 627.000000         | 627.000000 | 627.000000 |
| mean  | 29.631308  | 0.545455           | 0.379585   | 34.385399  |
| std   | 12.511818  | 1.151090           | 0.792999   | 54.597730  |
| min   | 0.750000   | 0.000000           | 0.000000   | 0.000000   |
| 25%   | 23.000000  | 0.000000           | 0.000000   | 7.895800   |
| 50%   | 28.000000  | 0.000000           | 0.000000   | 15.045800  |
| 75%   | 35.000000  | 1.000000           | 0.000000   | 31.387500  |
| max   | 80.000000  | 8.000000           | 5.000000   | 512.329200 |


训练和评估集分别有627和264个样本数据：

```python
dftrain.shape[0], dfeval.shape[0]
```

```
      (627, 264)
```

大多数乘客都在20和30年代

```python
dftrain.age.hist(bins=20)
```

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_15_1.png)


机上的男性乘客大约是女性乘客的两倍。

```python
dftrain.sex.value_counts().plot(kind='barh')
```

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_17_1.png)


大多数乘客都在“第三”阶级：

```python
dftrain['class'].value_counts().plot(kind='barh')
```

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_19_1.png)


与男性相比，女性的生存机会要高得多，这显然是该模型的预测特征：

```python
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
```

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_21_1.png)


## 5. 模型的特征工程

Estimator使用称为[特征列](https://www.tensorflow.org/guide/feature_columns)的系统来描述模型应如何解释每个原始输入特征，Estimator需要一个数字输入向量，而特征列描述模型应如何转换每个特征。

选择和制作正确的特征列是学习有效模型的关键，特征列可以是原始特征`dict`（基本特征列）中的原始输入之一，也可以是使用在一个或多个基本列（派生特征列）上定义的转换创建的任何新列。

线性Estimator同时使用数值和分类特征，特征列适用于所有TensorFlow Estimator，它们的目的是定义用于建模的特征。此外，它们还提供了一些特征工程功能，比如独热编码、归一化和分桶。


### 5.1. 基本特征列

```python
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
```

`input_function`指定如何将数据转换为以流方式提供输入管道的`tf.data.Dataset`。`tf.data.Dataset`采用多种来源，如数据帧DataFrame，csv格式的文件等。

```python
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```

检查数据集：

```python
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['class'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())
```

您还可以使用`tf.keras.layers.DenseFeatures`层检查特征列的结果：

```python
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()
```

```
      array([[38.],
             [39.],
             [28.],
             [28.],
             [36.],
             [71.],
             [24.],
             [47.],
             [23.],
             [28.]], dtype=float32)
```

`DenseFeatures`只接受密集张量，要检查分类列，需要先将其转换为指示列：

```python
gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()
```

```
      array([[0., 1.],
             [0., 1.],
             [1., 0.],
             [0., 1.],
             [1., 0.],
             [1., 0.],
             [1., 0.],
             [1., 0.],
             [1., 0.],
             [0., 1.]], dtype=float32)
```       

将所有基本特征添加到模型后，让我们训练模型。使用`tf.estimator` API训练模型只是一个命令：

```python
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)
```

```
        {'accuracy_baseline': 0.625, 'auc': 0.83722067, 'accuracy': 0.7462121, 'recall': 0.6666667, 'global_step': 200, 'prediction/mean': 0.38311505, 'average_loss': 0.47361037, 'precision': 0.66, 'auc_precision_recall': 0.7851523, 'loss': 0.46608958, 'label/mean': 0.375}
```

### 5.2. 派生特征列

现在你达到了75％的准确率。单独使用每个基本功能列可能不足以解释数据。例如，性别和标签之间的相关性可能因性别不同而不同。因此，如果您只学习`gender="Male"`和`gender="Female"`的单一模型权重，您将无法捕捉每个年龄-性别组合（例如，区分`gender="Male"`和`age="30"` 和`gender="Male"`和 `age="40"`）。

要了解不同特征组合之间的差异，可以将交叉特征列添加到模型中（也可以在交叉列之前对年龄进行分桶）：

```python
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
```

将组合特征添加到模型之后，让我们再次训练模型：

```python
derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)
```

```
      {'accuracy_baseline': 0.625, 'auc': 0.8424855, 'accuracy': 0.7689394, 'recall': 0.6060606, 'global_step': 200, 'prediction/mean': 0.30415845, 'average_loss': 0.49316654, 'precision': 0.73170733, 'auc_precision_recall': 0.7732599, 'loss': 0.48306185, 'label/mean': 0.375}
```      

它现在到达了77.6%的准确度，略好于仅在基本特征方面受过训练，您可以尝试使用更多特征和转换，看看您是否可以做得更好。

现在，您可以使用训练模型从评估集对乘客进行预测。TensorFlow模型经过优化，可以同时对样本的批处理或集合进行预测，之前的`eval_input_fn`是使用整个评估集定义的。

```python
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
```

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_42_1.png)

最后，查看结果的接收器操作特性（即ROC），这将使我们更好地了解真阳性率和假阳性率之间的权衡。

```python
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
```

`(0, 1.05)`

![png](https://tensorflow.google.cn/alpha/tutorials/estimators/linear_files/output_44_1.png)

