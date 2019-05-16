---
title: 预测燃油效率：回归
categories: tensorflow2.0官方文档
tags: tensorflow2.0
date: 2019-05-20
abbrlink: tensorflow/tensorflow2-tutorials-keras-basic_regression
---

# 预测燃油效率：回归

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/keras/basic_regression"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>


在*回归*问题中，我们的目标是预测连续值的输出，如价格或概率。
将此与*分类*问题进行对比，分类的目标是从类列表中选择一个类（例如，图片包含苹果或橙色，识别图片中的哪个水果）。

本章节采用了经典的[Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 数据集，并建立了一个模型来预测20世纪70年代末和80年代初汽车的燃油效率。为此，我们将为该模型提供该时段内许多汽车的描述，此描述包括以下属性：气缸，排量，马力和重量。

此实例使用tf.keras API，有关信息信息，请参阅[Keras指南](https://tensorflow.google.cn/guide/keras)。

```
# 使用seaborn进行pairplot数据可视化，安装命令
pip install seaborn
```


```
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# tensorflow2 安装命令 pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
```

## 1. Auto MPG数据集

该数据集可从[UCI机器学习库](https://archive.ics.uci.edu/ml/)获得。


### 1.1. 获取数据

首先下载数据集：

```
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path
```

用pandas导入数据

```
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
```

|     | MPG  | Cylinders | Displacement | Horsepower | Weight | Acceleration | Model Year | Origin |
|-----|------|-----------|--------------|------------|--------|--------------|------------|--------|
| 393 | 27.0 | 4         | 140.0        | 86.0       | 2790.0 | 15.6         | 82         | 1      |
| 394 | 44.0 | 4         | 97.0         | 52.0       | 2130.0 | 24.6         | 82         | 2      |
| 395 | 32.0 | 4         | 135.0        | 84.0       | 2295.0 | 11.6         | 82         | 1      |
| 396 | 28.0 | 4         | 120.0        | 79.0       | 2625.0 | 18.6         | 82         | 1      |
| 397 | 31.0 | 4         | 119.0        | 82.0       | 2720.0 | 19.4         | 82         | 1      |


### 1.2. 清理数据

数据集包含一些未知值

```
dataset.isna().sum()
```

```
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
```

这是一个入门教程，所以我们就简单地删除这些行。

```
dataset = dataset.dropna()
```

“Origin”这一列实际上是分类，而不是数字。 所以把它转换为独热编码：

```
origin = dataset.pop('Origin')
```

```
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()
```


|     | MPG  | Cylinders | Displacement | Horsepower | Weight | Acceleration | Model Year | USA | Europe | Japan |
|-----|------|-----------|--------------|------------|--------|--------------|------------|-----|--------|-------|
| 393 | 27.0 | 4         | 140.0        | 86.0       | 2790.0 | 15.6         | 82         | 1.0 | 0.0    | 0.0   |
| 394 | 44.0 | 4         | 97.0         | 52.0       | 2130.0 | 24.6         | 82         | 0.0 | 1.0    | 0.0   |
| 395 | 32.0 | 4         | 135.0        | 84.0       | 2295.0 | 11.6         | 82         | 1.0 | 0.0    | 0.0   |
| 396 | 28.0 | 4         | 120.0        | 79.0       | 2625.0 | 18.6         | 82         | 1.0 | 0.0    | 0.0   |
| 397 | 31.0 | 4         | 119.0        | 82.0       | 2720.0 | 19.4         | 82         | 1.0 | 0.0    | 0.0   |




### 1.3. 将数据分为训练集和测试集

现在将数据集拆分为训练集和测试集，我们将在模型的最终评估中使用测试集。

```
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```

### 1.4. 检查数据

快速浏览训练集中几对列的联合分布：

```
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_20_1.png)

另外查看整体统计数据：

```
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
```


|              | count | mean        | std        | min    | 25%     | 50%    | 75%     | max    |
|--------------|-------|-------------|------------|--------|---------|--------|---------|--------|
| Cylinders    | 314.0 | 5.477707    | 1.699788   | 3.0    | 4.00    | 4.0    | 8.00    | 8.0    |
| Displacement | 314.0 | 195.318471  | 104.331589 | 68.0   | 105.50  | 151.0  | 265.75  | 455.0  |
| Horsepower   | 314.0 | 104.869427  | 38.096214  | 46.0   | 76.25   | 94.5   | 128.00  | 225.0  |
| Weight       | 314.0 | 2990.251592 | 843.898596 | 1649.0 | 2256.50 | 2822.5 | 3608.00 | 5140.0 |
| Acceleration | 314.0 | 15.559236   | 2.789230   | 8.0    | 13.80   | 15.5   | 17.20   | 24.8   |
| Model Year   | 314.0 | 75.898089   | 3.675642   | 70.0   | 73.00   | 76.0   | 79.00   | 82.0   |
| USA          | 314.0 | 0.624204    | 0.485101   | 0.0    | 0.00    | 1.0    | 1.00    | 1.0    |
| Europe       | 314.0 | 0.178344    | 0.383413   | 0.0    | 0.00    | 0.0    | 0.00    | 1.0    |
| Japan        | 314.0 | 0.197452    | 0.398712   | 0.0    | 0.00    | 0.0    | 0.00    | 1.0    |


### 1.5. 从标签中分割特征

将目标值或“标签”与特征分开，此标签是您训练的模型进行预测的值：

```
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
```

### 1.6. 标准化数据

再看一下上面的`train_stats`块，注意每个特征的范围有多么不同。

使用不同的比例和范围对特征进行标准化是一个很好的实践，虽然模型可能在没有特征标准化的情况下收敛，但它使训练更加困难，并且它使得最终模型取决于输入中使用的单位的选择。

注意：尽管我们仅从训练数据集中有意生成这些统计信息，但这些统计信息也将用于标准化测试数据集。我们需要这样做，将测试数据集投影到模型已经训练过的相同分布中。

```
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
```

这个标准化数据是我们用来训练模型的数据。

注意：用于标准化输入的统计数据（平均值和标准偏差）需要应用于输入模型的任何其他数据，以及我们之前执行的独热编码。这包括测试集以及模型在生产中使用时的实时数据。

## 2. 模型

### 2.1. 构建模型

让我们建立我们的模型。在这里，我们将使用具有两个密集连接隐藏层的`Sequential`模型，以及返回单个连续值的输出层。模型构建步骤包含在函数`build_model`中，因为我们稍后将创建第二个模型。

```
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```


```
model = build_model()
```

### 2.2. 检查模型

使用`.summary`方法打印模型的简单描述

```
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                640       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 4,865
Trainable params: 4,865
Non-trainable params: 0
_________________________________________________________________
```

现在试试这个模型。从训练数据中取出一批10个样本数据并在调用`model.predict`函数。

```
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result
```

```
array([[ 0.3297699 ],
       [ 0.25655937],
       [-0.12460149],
       [ 0.32495883],
       [ 0.50459725],
       [ 0.10887371],
       [ 0.57305855],
       [ 0.57637435],
       [ 0.12094647],
       [ 0.6864784 ]], dtype=float32)
```

这似乎可以工作，它产生预期的shape和类型的结果。

### 2.3. 训练模型

训练模型1000个周期，并在`history`对象中记录训练和验证准确性：

```
# 通过为每个完成的周期打印单个点来显示训练进度 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
```

使用存储在`history`对象中的统计数据可视化模型的训练进度。300

```
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
```

|     | loss     | mae      | mse      | val_loss  | val_mae  | val_mse   | epoch |
|-----|----------|----------|----------|-----------|----------|-----------|-------|
| 995 | 2.556746 | 0.988013 | 2.556746 | 10.210531 | 2.324411 | 10.210530 | 995   |
| 996 | 2.597973 | 1.039339 | 2.597973 | 11.257273 | 2.469266 | 11.257273 | 996   |
| 997 | 2.671929 | 1.040886 | 2.671929 | 10.604957 | 2.446257 | 10.604958 | 997   |
| 998 | 2.634858 | 1.001898 | 2.634858 | 10.906935 | 2.373279 | 10.906935 | 998   |
| 999 | 2.741717 | 1.035889 | 2.741717 | 10.698320 | 2.342703 | 10.698319 | 999   |

```
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)
```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_42_0.png?dcb_=0.7319815786783315)

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_42_1.png?dcb_=0.09774210050560783)


该图表显示在约100个周期之后，验证误差几乎没有改进，甚至降低。让我们更新`model.fit`调用，以便在验证分数没有提高时自动停止训练。我们将使用`EarlyStopping`回调来测试每个周期的训练状态。如果经过一定数量的周期而没有显示出改进，则自动停止训练。


您可以了解此回调的更多信息 [连接](https://tensorflow.google.cn/versions/master/api_docs/python/tf/keras/callbacks/EarlyStopping).

```
model = build_model()

# “patience”参数是检查改进的周期量 
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_44_1.png?dcb_=0.8643233947217597)

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_44_2.png?dcb_=0.8788778722328034)

The graph shows that on the validation set, the average error is usually around +/- 2 MPG. Is this good? We'll leave that decision up to you.

Let's see how well the model generalizes by using the **test** set, which we did not use when training the model.  This tells us how well we can expect the model to predict when we use it in the real world.
上图显示在验证集上平均误差通常约为+/-2MPG，这个好吗？我们会把这个决定留给你。

让我们看一下使用测试集来看一下泛化模型效果，我们在训练模型时没有使用测试集，这告诉我们，当我们在现实世界中使用模型时，我们可以期待模型预测。

```
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
```

`Testing set Mean Abs Error:  2.09 MPG`

### 2.4. 预测

最后，使用测试集中的数据预测MPG值：

```
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_48_0.png?dcb_=0.5259404035812005)

看起来我们的模型预测得相当好，我们来看看错误分布：

```
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
```

![png](https://tensorflow.google.cn/alpha/tutorials/keras/basic_regression_files/output_50_0.png?dcb_=0.042220469967213514)

上图看起来不是很高斯（正态分布），很可能是因为样本数据非常少。

## 3. 结论

本章节介绍了一些处理回归问题的技巧：

* 均方误差（MSE）是用于回归问题的常见损失函数（不同的损失函数用于分类问题）。

* 同样，用于回归的评估指标与分类不同，常见的回归度量是平均绝对误差（MAE）。

* 当数字输入数据特征具有不同范围的值时，应将每个特征独立地缩放到相同范围。

* 如果没有太多训练数据，应选择隐藏层很少的小网络，以避免过拟合。

* 尽早停止是防止过拟合的有效技巧。
