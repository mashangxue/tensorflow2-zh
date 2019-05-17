
# 自定义训练：演示

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/eager/custom_training_walkthrough"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/eager/custom_training_walkthrough.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/eager/custom_training_walkthrough.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

本指南使用机器学习按品种对鸢尾花进行分类。它利用 TensorFlow 的 Eager Execution 来执行以下操作： 

1. 构建模型

2. 使用样本数据训练该模型

3. 利用该模型对未知数据进行预测。

## TensorFlow 编程

本指南采用了以下高级 TensorFlow 概念：

* 使用TensorFlow的默认  [eager execution](https://www.tensorflow.org/guide/eager) 开发环境,

* 使用 [Datasets API](https://www.tensorflow.org/guide/datasets) 导入数据，

* 使用 TensorFlow 的 [Keras API](https://keras.io/getting-started/sequential-model-guide/) 构建模型和层。

本教程采用了与许多 TensorFlow 程序相似的结构：

1. 导入和解析数据集。

2. 选择模型类型。

3. 训练模型。

4. 评估模型的效果。

5. 使用经过训练的模型进行预测。

## 设置程序

### 配置导入

导入所需的 Python 模块（包括 TensorFlow），默认情况下，TensorFlow使用 Eager Execution 来立即评估操作，并返回具体的值，而不是创建稍后执行的计算图。如果您习惯使用 REPL 或 python 交互控制台，对于 Eager Execution 您会用起来得心应手。

```
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt

import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
```

```
      TensorFlow version: 2.0.0-alpha0 Eager execution: True
```

## 鸢尾花分类问题 The Iris classification problem

想象一下，您是一名植物学家，正在寻找一种能够对所发现的每株鸢尾花进行自动归类的方法。机器学习可提供多种从统计学上分类花卉的算法。例如，一个复杂的机器学习程序可以根据照片对花卉进行分类。我们的要求并不高，我们将根据鸢尾花花萼和花瓣的长度和宽度对其进行分类。

鸢尾属约有 300 个品种，但我们的程序将仅对下列三个品种进行分类：

* 山鸢尾
* 维吉尼亚鸢尾
* 变色鸢尾

<table>
  <tr><td>
    <img src="https://tensorflow.google.cn/images/iris_three_species.jpg"
         alt="Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor">
  </td></tr>
  <tr><td align="center">
    <b>图1.</b> <a href="https://commons.wikimedia.org/w/index.php?curid=170298">山鸢尾Iris setosa, <a href="https://commons.wikimedia.org/w/index.php?curid=248095">变色鸢尾Iris versicolor</a>，和 <a href="https://www.flickr.com/photos/33397993@N05/3352169862">维吉尼亚鸢尾Iris virginica</a> <br/>&nbsp;
  </td></tr>
</table>

幸运的是，有人已经创建了一个包含 120 株鸢尾花的数据集（其中有花萼和花瓣的测量值）。这是一个在入门级机器学习分类问题中经常使用的经典数据集。

## 导入和解析训练数据集

下载数据集文件并将其转换为可供此 Python 程序使用的结构。

### 下载数据集

使用 [tf.keras.utils.get_file](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file) 函数下载训练数据集文件。该函数会返回下载文件的文件路径。

```python
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
```

### 检查数据 Inspect the data

数据集 `iris_training.csv` 是一个纯文本文件，其中存储了逗号分隔值 (CSV) 格式的表格式数据。请使用 `head -n5` 命令查看前 5 个条目：

```
!head -n5 {train_dataset_fp}
```

```
      120,4,setosa,versicolor,virginica 
      6.4,2.8,5.6,2.2,2 
      5.0,2.3,3.3,1.0,1 
      4.9,2.5,4.5,1.7,2 
      4.9,3.1,1.5,0.1,0
```

我们可以从该数据集视图中注意到以下信息：

1. 第一行是标题，其中包含数据集信息：
* 共有 120 个样本。每个样本都有四个特征和一个标签名称，标签名称有三种可能。

2. 后面的行是数据记录，每个样本各占一行，其中：
* 前四个字段是特征：即样本的特点。在此数据集中，这些字段存储的是代表花卉测量值的浮点数。
* 最后一列是标签：即我们想要预测的值。对于此数据集，该值为 0、1 或 2 中的某个整数值（每个值分别对应一个花卉名称）。

我们用代码表示出来：

```
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))
```

```
      Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] Label: species
```

每个标签都分别与一个字符串名称（例如“setosa”）相关联，但机器学习通常依赖于数字值。标签编号会映射到一个指定的表示法，例如：

* `0`: 山鸢尾
* `1`: 变色鸢尾
* `2`: 维吉尼亚鸢尾

如需详细了解特征和标签，请参阅[《机器学习速成课程》的“机器学习术语”部分](https://developers.google.cn/machine-learning/crash-course/framing/ml-terminology)。

```
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
```

### 创建一个 `tf.data.Dataset`

TensorFlow's [Dataset API](https://www.tensorflow.org/guide/datasets) handles many common cases for loading data into a model. This is a high-level API for reading data and transforming it into a form used for training. See the [Datasets Quick Start guide](https://www.tensorflow.org/get_started/datasets_quickstart) for more information.


Since the dataset is a CSV-formatted text file, use the [make_csv_dataset](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset) function to parse the data into a suitable format. Since this function generates data for training models, the default behavior is to shuffle the data (`shuffle=True, shuffle_buffer_size=10000`), and repeat the dataset forever (`num_epochs=None`). We also set the [batch_size](https://developers.google.com/machine-learning/glossary/#batch_size) parameter.

TensorFlow 的 Dataset API 可处理在向模型加载数据时遇到的许多常见情况。这是一种高阶 API，用于读取数据并将其转换为可供训练使用的格式。如需了解详情，请参阅[数据集快速入门指南](https://tensorflow.google.cn/guide/datasets_for_estimators)。

由于数据集是 CSV 格式的文本文件，请使用 make_csv_dataset 函数将数据解析为合适的格式。由于此函数为训练模型生成数据，默认行为是对数据进行随机处理 (`shuffle=True, shuffle_buffer_size=10000`)，并且无限期重复数据集 (`num_epochs=None`)。我们还设置了 batch_size 参数。

```
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
```

The `make_csv_dataset` function returns a `tf.data.Dataset` of `(features, label)` pairs, where `features` is a dictionary: `{'feature_name': value}`

These `Dataset` objects are iterable. Let's look at a batch of features:

`make_csv_dataset` 函数返回 `(features, label)` 对的 `tf.data.Dataset`，其中 `features` 是一个字典：`{'feature_name': value}`

这些 Dataset 对象便可迭代。我们来看看一批特征：

```
features, labels = next(iter(train_dataset))

print(features)
```

Notice that like-features are grouped together, or *batched*. Each example row's fields are appended to the corresponding feature array. Change the `batch_size` to set the number of examples stored in these feature arrays.

You can start to see some clusters by plotting a few features from the batch:

请注意，类似特征会归为一组，即分为一批。每个样本行的字段都会附加到相应的特征数组中。更改 batch_size 可设置存储在这些特征数组中的样本数。

绘制该批次中的几个特征后，就会开始看到一些集群现象：

```
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
```

![png](https://tensorflow.google.cn/alpha/tutorials/eager/custom_training_walkthrough_files/output_22_0.png)

To simplify the model building step, create a function to repackage the features dictionary into a single array with shape: `(batch_size, num_features)`.

This function uses the [tf.stack](https://www.tensorflow.org/api_docs/python/tf/stack) method which takes values from a list of tensors and creates a combined tensor at the specified dimension.

要简化模型构建步骤，请创建一个函数以将特征字典重新打包为形状为 `(batch_size, num_features)` 的单个数组。

此函数使用 [tf.stack](https://tensorflow.google.cn/api_docs/python/tf/stack) 方法，该方法从张量列表中获取值，并创建指定维度的组合张量。

```
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
```

Then use the [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/dataset/map) method to pack the `features` of each `(features,label)` pair into the training dataset:

然后使用 [tf.data.Dataset.map](https://tensorflow.google.cn/api_docs/python/tf/data/dataset/map) 方法将每个 `(features,label)` 对的 `features` 打包到训练数据集中：

```
train_dataset = train_dataset.map(pack_features_vector)
```

The features element of the `Dataset` are now arrays with shape `(batch_size, num_features)`. Let's look at the first few examples:

`Dataset` 的 features 元素现在是形状为 `(batch_size, num_features)` 的数组。我们来看看前几个样本：

```
features, labels = next(iter(train_dataset))

print(features[:5])
```

```
    tf.Tensor( 
    [[4.9 2.4 3.3 1. ] 
    ...
    [6.6 3. 4.4 1.4]], shape=(5, 4), dtype=float32)
```

## 选择模型类型 Select the type of model

### 为何要使用模型？ Why model?

A *[model](https://developers.google.com/machine-learning/crash-course/glossary#model)* is a relationship between features and the label.  For the Iris classification problem, the model defines the relationship between the sepal and petal measurements and the predicted Iris species. Some simple models can be described with a few lines of algebra, but complex machine learning models have a large number of parameters that are difficult to summarize.

Could you determine the relationship between the four features and the Iris species *without* using machine learning?  That is, could you use traditional programming techniques (for example, a lot of conditional statements) to create a model?  Perhaps—if you analyzed the dataset long enough to determine the relationships between petal and sepal measurements to a particular species. And this becomes difficult—maybe impossible—on more complicated datasets. A good machine learning approach *determines the model for you*. If you feed enough representative examples into the right machine learning model type, the program will figure out the relationships for you.

模型是指特征与标签之间的关系。对于鸢尾花分类问题，模型定义了花萼和花瓣测量值与预测的鸢尾花品种之间的关系。一些简单的模型可以用几行代数进行描述，但复杂的机器学习模型拥有大量难以汇总的参数。

您能否在不使用机器学习的情况下确定四个特征与鸢尾花品种之间的关系？也就是说，您能否使用传统编程技巧（例如大量条件语句）创建模型？也许能，前提是反复分析该数据集，并最终确定花瓣和花萼测量值与特定品种的关系。对于更复杂的数据集来说，这会变得非常困难，或许根本就做不到。一个好的机器学习方法可为您确定模型。如果您将足够多的代表性样本馈送到正确类型的机器学习模型中，该程序便会为您找出相应的关系。

### 选择模型 Select the model

We need to select the kind of model to train. There are many types of models and picking a good one takes experience. This tutorial uses a neural network to solve the Iris classification problem. *[Neural networks](https://developers.google.com/machine-learning/glossary/#neural_network)* can find complex relationships between features and the label. It is a highly-structured graph, organized into one or more *[hidden layers](https://developers.google.com/machine-learning/glossary/#hidden_layer)*. Each hidden layer consists of one or more *[neurons](https://developers.google.com/machine-learning/glossary/#neuron)*. There are several categories of neural networks and this program uses a dense, or *[fully-connected neural network](https://developers.google.com/machine-learning/glossary/#fully_connected_layer)*: the neurons in one layer receive input connections from *every* neuron in the previous layer. For example, Figure 2 illustrates a dense neural network consisting of an input layer, two hidden layers, and an output layer:

我们需要选择要进行训练的模型类型。模型具有许多类型，挑选合适的类型需要一定的经验。本教程使用神经网络来解决鸢尾花分类问题。神经网络可以发现特征与标签之间的复杂关系。神经网络是一个高度结构化的图，其中包含一个或多个隐藏层。每个隐藏层都包含一个或多个神经元。神经网络有多种类别，该程序使用的是密集型神经网络，也称为全连接神经网络：一个层中的神经元将从上一层中的每个神经元获取输入连接。例如，图 2 显示了一个密集型神经网络，其中包含 1 个输入层、2 个隐藏层以及 1 个输出层：

<table>
  <tr><td>
    <img src="https://tensorflow.google.cn/images/custom_estimators/full_network.png"
         alt="A diagram of the network architecture: Inputs, 2 hidden layers, and outputs">
  </td></tr>
  <tr><td align="center">
    <b>图 2.</b> 包含特征、隐藏层和预测的神经网络 <br/>&nbsp;
  </td></tr>
</table>

When the model from Figure 2 is trained and fed an unlabeled example, it yields three predictions: the likelihood that this flower is the given Iris species. This prediction is called *[inference](https://developers.google.com/machine-learning/crash-course/glossary#inference)*. For this example, the sum of the output predictions is 1.0. In Figure 2, this prediction breaks down as: `0.02` for *Iris setosa*, `0.95` for *Iris versicolor*, and `0.03` for *Iris virginica*. This means that the model predicts—with 95% probability—that an unlabeled example flower is an *Iris versicolor*.

当图 2 中的模型经过训练并馈送未标记的样本时，它会产生 3 个预测结果：相应鸢尾花属于指定品种的可能性。这种预测称为[推理](https://developers.google.cn/machine-learning/crash-course/glossary#inference)。对于该示例，输出预测结果的总和是 1.0。在图 2 中，该预测结果分解如下：山鸢尾为 0.02，变色鸢尾为 0.95，维吉尼亚鸢尾为 0.03。这意味着该模型预测某个无标签鸢尾花样本是变色鸢尾的概率为 95％。


### 使用Keras创建模型 Create a model using Keras

The TensorFlow [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) API is the preferred way to create models and layers. This makes it easy to build models and experiment while Keras handles the complexity of connecting everything together.

The [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) model is a linear stack of layers. Its constructor takes a list of layer instances, in this case, two [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layers with 10 nodes each, and an output layer with 3 nodes representing our label predictions. The first layer's `input_shape` parameter corresponds to the number of features from the dataset, and is required.

TensorFlow `tf.keras` API 是创建模型和层的首选方式。通过该 API，您可以轻松地构建模型并进行实验，而将所有部分连接在一起的复杂工作则由 Keras 处理。

`tf.keras.Sequential` 模型是层的线性堆叠。该模型的构造函数会采用一系列层实例；在本示例中，采用的是 2 个密集层（分别包含 10 个节点）以及 1 个输出层（包含 3 个代表标签预测的节点）。第一个层的 `input_shape` 参数对应该数据集中的特征数量，它是一项必需参数。

```
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
```

The *[activation function](https://developers.google.com/machine-learning/crash-course/glossary#activation_function)* determines the output shape of each node in the layer. These non-linearities are important—without them the model would be equivalent to a single layer. There are many [available activations](https://www.tensorflow.org/api_docs/python/tf/keras/activations), but [ReLU](https://developers.google.com/machine-learning/crash-course/glossary#ReLU) is common for hidden layers.

The ideal number of hidden layers and neurons depends on the problem and the dataset. Like many aspects of machine learning, picking the best shape of the neural network requires a mixture of knowledge and experimentation. As a rule of thumb, increasing the number of hidden layers and neurons typically creates a more powerful model, which requires more data to train effectively.

[激活函数](https://developers.google.cn/machine-learning/crash-course/glossary#activation_function)可决定层中每个节点的输出形状。这些非线性关系很重要，如果没有它们，模型将等同于单个层。[激活函数有很多](https://tensorflow.google.cn/api_docs/python/tf/keras/activations)，但隐藏层通常使用 [ReLU](https://developers.google.cn/machine-learning/crash-course/glossary#ReLU)。

隐藏层和神经元的理想数量取决于问题和数据集。与机器学习的多个方面一样，选择最佳的神经网络形状需要一定的知识水平和实验基础。一般来说，增加隐藏层和神经元的数量通常会产生更强大的模型，而这需要更多数据才能有效地进行训练。

### 使用模型  Using the model

Let's have a quick look at what this model does to a batch of features:

我们快速了解一下此模型如何处理一批特征：

```
predictions = model(features)
predictions[:5]
```

Here, each example returns a [logit](https://developers.google.com/machine-learning/crash-course/glossary#logits) for each class.

To convert these logits to a probability for each class, use the [softmax](https://developers.google.com/machine-learning/crash-course/glossary#softmax) function:

在此示例中，每个样本针对每个类别返回一个[logit](https://developers.google.cn/machine-learning/crash-course/glossary#logits) 。

要将这些对数转换为每个类别的概率，请使用 [softmax](https://developers.google.cn/machine-learning/crash-course/glossary#softmax)  函数：

```
tf.nn.softmax(predictions[:5])
```

Taking the `tf.argmax` across classes gives us the predicted class index. But, the model hasn't been trained yet, so these aren't good predictions.

对每个类别执行 `tf.argmax` 运算可得出预测的类别索引。不过，该模型尚未接受训练，因此这些预测并不理想。

```
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))
```

## 训练模型 Train the model

*[Training](https://developers.google.com/machine-learning/crash-course/glossary#training)* is the stage of machine learning when the model is gradually optimized, or the model *learns* the dataset. The goal is to learn enough about the structure of the training dataset to make predictions about unseen data. If you learn *too much* about the training dataset, then the predictions only work for the data it has seen and will not be generalizable. This problem is called *[overfitting](https://developers.google.com/machine-learning/crash-course/glossary#overfitting)*—it's like memorizing the answers instead of understanding how to solve a problem.

The Iris classification problem is an example of *[supervised machine learning](https://developers.google.com/machine-learning/glossary/#supervised_machine_learning)*: the model is trained from examples that contain labels. In *[unsupervised machine learning](https://developers.google.com/machine-learning/glossary/#unsupervised_machine_learning)*, the examples don't contain labels. Instead, the model typically finds patterns among the features.

训练是一个机器学习阶段，在此阶段中，模型会逐渐得到优化，也就是说，模型会了解数据集。目标是充分了解训练数据集的结构，以便对未见过的数据进行预测。如果您从训练数据集中获得了过多的信息，预测便会仅适用于模型见过的数据，但是无法泛化。此问题称为[过拟合](https://developers.google.cn/machine-learning/crash-course/glossary#overfitting)，好比将答案死记硬背下来，而不去理解问题的解决方式。

鸢尾花分类问题是监督式机器学习的一个示例：模型通过包含标签的样本加以训练。在非监督式机器学习中，样本不包含标签。相反，模型通常会在特征中发现一些规律。

### 定义损失和梯度函数 Define the loss and gradient function

Both training and evaluation stages need to calculate the model's *[loss](https://developers.google.com/machine-learning/crash-course/glossary#loss)*. This measures how off a model's predictions are from the desired label, in other words, how bad the model is performing. We want to minimize, or optimize, this value.

Our model will calculate its loss using the `tf.keras.losses.SparseCategoricalCrossentropy` function which takes the model's class probability predictions and the desired label, and returns the average loss across the examples.

在训练和评估阶段，我们都需要计算模型的损失。这样可以衡量模型的预测结果与预期标签有多大偏差，也就是说，模型的效果有多差。我们希望尽可能减小或优化这个值。

我们的模型会使用 `tf.keras.losses.SparseCategoricalCrossentropy` 函数计算其损失，此函数会接受模型的类别概率预测结果和预期标签，然后返回样本的平均损失。

```
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```


```
def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))
```

Use the [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) context to calculate the *[gradients](https://developers.google.com/machine-learning/crash-course/glossary#gradient)* used to optimize our model.

使用 [tf.GradientTape](https://tensorflow..google.cn/api_docs/python/tf/GradientTape) 上下计算用于优化模型的[梯度](https://developers.google.cn/machine-learning/crash-course/glossary#gradient)。

```
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
```

### 创建优化器 Create an optimizer

An *[optimizer](https://developers.google.com/machine-learning/crash-course/glossary#optimizer)* applies the computed gradients to the model's variables to minimize the `loss` function. You can think of the loss function as a curved surface (see Figure 3) and we want to find its lowest point by walking around. The gradients point in the direction of steepest ascent—so we'll travel the opposite way and move down the hill. By iteratively calculating the loss and gradient for each batch, we'll adjust the model during training. Gradually, the model will find the best combination of weights and bias to minimize loss. And the lower the loss, the better the model's predictions.

*[优化器](https://developers.google.cn/machine-learning/crash-course/glossary#optimizer)* 会将计算出的梯度应用于模型的变量，以最小化 loss 函数。您可以将损失函数想象为一个曲面（见图 3），我们希望通过到处走动找到该曲面的最低点。梯度指向最高速上升的方向，因此我们将沿相反的方向向下移动。我们以迭代方式计算每个批次的损失和梯度，以在训练过程中调整模型。模型会逐渐找到权重和偏差的最佳组合，从而将损失降至最低。损失越低，模型的预测效果就越好。


<table>
  <tr><td>
    <img src="https://cs231n.github.io/assets/nn3/opt1.gif" width="70%"
         alt="Optimization algorithms visualized over time in 3D space.">
  </td></tr>
  <tr><td align="center">
    <b>图 3.</b> 优化算法在三维空间中随时间推移而变化的可视化效果。<br/>(Source: <a href="http://cs231n.github.io/neural-networks-3/">斯坦福大学 CS231n 课程</a>, MIT License, Image credit: <a href="https://twitter.com/alecrad">Alec Radford</a>)
  </td></tr>
</table>

TensorFlow has many [optimization algorithms](https://www.tensorflow.org/api_guides/python/train) available for training. This model uses the [tf.train.GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer) that implements the *[stochastic gradient descent](https://developers.google.com/machine-learning/crash-course/glossary#gradient_descent)* (SGD) algorithm. The `learning_rate` sets the step size to take for each iteration down the hill. This is a *hyperparameter* that you'll commonly adjust to achieve better results.

Let's setup the optimizer:

TensorFlow 拥有许多可用于训练的[优化算法](https://www.tensorflow.org/api_guides/python/train)。此模型使用的是 [tf.train.GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)，它可以实现[随机梯度下降法](https://developers.google.cn/machine-learning/crash-course/glossary#gradient_descent) (SGD)。`learning_rate` 用于设置每次迭代（向下行走）的步长。这是一个超参数，您通常需要调整此参数以获得更好的结果。

我们来设置优化器：

```
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
```

We'll use this to calculate a single optimization step:

我们将使用它来计算单个优化步骤：

```
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels).numpy()))
```

```
      Step: 0, Initial Loss: 2.3108744621276855 
      Step: 1, Loss: 1.7618987560272217
```

### 训练循环 Training loop

With all the pieces in place, the model is ready for training! A training loop feeds the dataset examples into the model to help it make better predictions. The following code block sets up these training steps:

1. Iterate each *epoch*. An epoch is one pass through the dataset.
2. Within an epoch, iterate over each example in the training `Dataset` grabbing its *features* (`x`) and *label* (`y`).
3. Using the example's features, make a prediction and compare it with the label. Measure the inaccuracy of the prediction and use that to calculate the model's loss and gradients.
4. Use an `optimizer` to update the model's variables.
5. Keep track of some stats for visualization.
6. Repeat for each epoch.

The `num_epochs` variable is the number of times to loop over the dataset collection. Counter-intuitively, training a model longer does not guarantee a better model. `num_epochs` is a *[hyperparameter](https://developers.google.com/machine-learning/glossary/#hyperparameter)* that you can tune. Choosing the right number usually requires both experience and experimentation.

一切准备就绪后，就可以开始训练模型了！训练循环会将数据集样本馈送到模型中，以帮助模型做出更好的预测。以下代码块可设置这些训练步骤：

1. 迭代每个周期。通过一次数据集即为一个周期。
2. 在一个周期中，遍历训练 Dataset 中的每个样本，并获取样本的特征 (x) 和标签 (y)。
3. 根据样本的特征进行预测，并比较预测结果和标签。衡量预测结果的不准确性，并使用所得的值计算模型的损失和梯度。
4. 使用 optimizer 更新模型的变量。
5. 跟踪一些统计信息以进行可视化。
6. 对每个周期重复执行以上步骤。

num_epochs 变量是遍历数据集集合的次数。与直觉恰恰相反的是，训练模型的时间越长，并不能保证模型就越好。num_epochs 是一个可以调整的超参数。选择正确的次数通常需要一定的经验和实验基础。

```
## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
```

```
      Epoch 000: Loss: 1.568, Accuracy: 30.000% 
      Epoch 050: Loss: 0.061, Accuracy: 98.333% 
      Epoch 100: Loss: 0.058, Accuracy: 97.500% 
      Epoch 150: Loss: 0.044, Accuracy: 99.167% 
      Epoch 200: Loss: 0.049, Accuracy: 97.500%
```

### 可视化损失函数随时间推移而变化的情况 Visualize the loss function over time

While it's helpful to print out the model's training progress, it's often *more* helpful to see this progress. [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is a nice visualization tool that is packaged with TensorFlow, but we can create basic charts using the `matplotlib` module.

Interpreting these charts takes some experience, but you really want to see the *loss* go down and the *accuracy* go up.

虽然输出模型的训练过程有帮助，但查看这一过程往往更有帮助。TensorBoard 是与 TensorFlow 封装在一起的出色可视化工具，不过我们可以使用 matplotlib 模块创建基本图表。

解读这些图表需要一定的经验，不过您确实希望看到损失下降且准确率上升。

```
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
```

![png](https://tensorflow.google.cn/alpha/tutorials/eager/custom_training_walkthrough_files/output_54_0.png)

## 评估模型的效果 Evaluate the model's effectiveness

Now that the model is trained, we can get some statistics on its performance.

*Evaluating* means determining how effectively the model makes predictions. To determine the model's effectiveness at Iris classification, pass some sepal and petal measurements to the model and ask the model to predict what Iris species they represent. Then compare the model's prediction against the actual label.  For example, a model that picked the correct species on half the input examples has an *[accuracy](https://developers.google.com/machine-learning/glossary/#accuracy)* of `0.5`. Figure 4 shows a slightly more effective model, getting 4 out of 5 predictions correct at 80% accuracy:

模型已经过训练，现在我们可以获取一些关于其效果的统计信息了。

评估指的是确定模型做出预测的效果。要确定模型在鸢尾花分类方面的效果，请将一些花萼和花瓣测量值传递给模型，并要求模型预测它们所代表的鸢尾花品种。然后，将模型的预测结果与实际标签进行比较。例如，如果模型对一半输入样本的品种预测正确，则准确率为 0.5。图 4 显示的是一个效果更好一些的模型，该模型做出 5 次预测，其中有 4 次正确，准确率为 80%：

<table cellpadding="8" border="0">
  <colgroup>
    <col span="4" >
    <col span="1" bgcolor="lightblue">
    <col span="1" bgcolor="lightgreen">
  </colgroup>
  <tr bgcolor="lightgray">
    <th colspan="4">样本特征</th>
    <th colspan="1">标签</th>
    <th colspan="1" >模型预测</th>
  </tr>
  <tr>
    <td>5.9</td><td>3.0</td><td>4.3</td><td>1.5</td><td align="center">1</td><td align="center">1</td>
  </tr>
  <tr>
    <td>6.9</td><td>3.1</td><td>5.4</td><td>2.1</td><td align="center">2</td><td align="center">2</td>
  </tr>
  <tr>
    <td>5.1</td><td>3.3</td><td>1.7</td><td>0.5</td><td align="center">0</td><td align="center">0</td>
  </tr>
  <tr>
    <td>6.0</td> <td>3.4</td> <td>4.5</td> <td>1.6</td> <td align="center">1</td><td align="center" bgcolor="red">2</td>
  </tr>
  <tr>
    <td>5.5</td><td>2.5</td><td>4.0</td><td>1.3</td><td align="center">1</td><td align="center">1</td>
  </tr>
  <tr><td align="center" colspan="6">
    <b>图4.</b> 准确率为 80% 的鸢尾花分类器。<br/>&nbsp;
  </td></tr>
</table>

### 设置测试数据集 Setup the test dataset

Evaluating the model is similar to training the model. The biggest difference is the examples come from a separate *[test set](https://developers.google.com/machine-learning/crash-course/glossary#test_set)* rather than the training set. To fairly assess a model's effectiveness, the examples used to evaluate a model must be different from the examples used to train the model.

The setup for the test `Dataset` is similar to the setup for training `Dataset`. Download the CSV text file and parse that values, then give it a little shuffle:

评估模型与训练模型相似。最大的区别在于，样本来自一个单独的测试集，而不是训练集。为了公正地评估模型的效果，用于评估模型的样本务必与用于训练模型的样本不同。

测试 Dataset 的设置与训练 Dataset 的设置相似。下载 CSV 文本文件并解析相应的值，然后对数据稍加随机化处理：

```
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
```


```
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)
```

### 根据测试数据集评估模型 Evaluate the model on the test dataset

Unlike the training stage, the model only evaluates a single [epoch](https://developers.google.com/machine-learning/glossary/#epoch) of the test data. In the following code cell, we iterate over each example in the test set and compare the model's prediction against the actual label. This is used to measure the model's accuracy across the entire test set.

与训练阶段不同，模型仅评估测试数据的一个周期。在以下代码单元格中，我们会遍历测试集中的每个样本，然后将模型的预测结果与实际标签进行比较。这是为了衡量模型在整个测试集中的准确率。

```
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
```
```
      Test set accuracy: 96.667%
```

We can see on the last batch, for example, the model is usually correct:

例如，我们可以看到对于最后一批数据，该模型通常预测正确：

```
tf.stack([y,prediction],axis=1)
```

```
      <tf.Tensor: id=164408, shape=(30, 2), dtype=int32, numpy= 
      array([[1, 1], 
             [2, 2], 
             [0, 0],..., dtype=int32)>
```

## 使用经过训练的模型进行预测 Use the trained model to make predictions

We've trained a model and "proven" that it's good—but not perfect—at classifying Iris species. Now let's use the trained model to make some predictions on [unlabeled examples](https://developers.google.com/machine-learning/glossary/#unlabeled_example); that is, on examples that contain features but not a label.

In real-life, the unlabeled examples could come from lots of different sources including apps, CSV files, and data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels. Recall, the label numbers are mapped to a named representation as:

* `0`: Iris setosa
* `1`: Iris versicolor
* `2`: Iris virginica

我们已经训练了一个模型并“证明”它是有效的，但在对鸢尾花品种进行分类方面，这还不够。现在，我们使用经过训练的模型对无标签样本（即包含特征但不包含标签的样本）进行一些预测。

在现实生活中，无标签样本可能来自很多不同的来源，包括应用程序、CSV 文件和数据源。暂时我们将手动提供三个无标签样本以预测其标签。回想一下，标签编号会映射到一个指定的表示法：

* `0`：山鸢尾
* `1`：变色鸢尾
* `2`：维吉尼亚鸢尾

```
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
```

```
      Example 0 prediction: Iris setosa (100.0%)
      Example 1 prediction: Iris versicolor (100.0%) 
      Example 2 prediction: Iris virginica (99.5%)
```
