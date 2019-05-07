

# 电影评论的文字分类

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/keras/basic_text_classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_text_classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_text_classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>


本章节使用评论文本将电影评论分类为正面或负面。这是二元或两类分类的一个例子，是一种重要且广泛使用的机器学习问题。

我们将使用包含来自[网络电影数据库](https://www.imdb.com/)的50,000条电影评论文本的[IMDB数据集](https://tensorflow.google.cn/api_docs/python/tf/keras/datasets/imdb)，这些被分为25,000条训练评论和25,000条评估评论，训练和测试集是平衡的，这意味着它们包含相同数量的正面和负面评论。

本章节使用tf.keras，这是一个高级API，用于在TensorFlow中构建和训练模型，有关使用tf.keras的更高级文本分类教程，请参阅[MLCC文本分类指南](https://developers.google.cn/machine-learning/guides/text-classification/)。

```
from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tf-nightly-2.0-preview
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
```
`2.0.0-alpha0`                                                        
## 下载IMDB数据集

IMDB数据集与TensorFlow一起打包，它已经被预处理，使得评论（单词序列）已被转换为整数序列，其中每个整数表示字典中的特定单词。

以下代码将IMDB数据集下载到您的计算机（如果您已经下载了它，则使用缓存副本）：

```
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```  

`Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz 17465344/17464789 [==============================] - 0s 0us/step`

参数 `num_words=10000` 保留训练数据中最常出现的10,000个单词，丢弃罕见的单词以保持数据的大小可管理。

## 探索数据

我们花一点时间来理解数据的格式，数据集经过预处理：每个示例都是一个整数数组，表示电影评论的单词。每个标签都是0或1的整数值，其中0表示负面评论，1表示正面评论。

```
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
```

`Training entries: 25000, labels: 25000`

评论文本已转换为整数，其中每个整数表示字典中的特定单词。以下是第一篇评论的内容：

```
print(train_data[0])
```
`[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]`

电影评论的长度可能不同，以下代码显示了第一次和第二次评论中的字数。由于对神经网络的输入必须是相同的长度，我们稍后需要解决此问题。

```
len(train_data[0]), len(train_data[1])
```
*(218, 189)*

### 将整数转换成文本

了解如何将整数转换回文本可能很有用。
在这里，我们将创建一个辅助函数来查询包含整数到字符串映射的字典对象：

```
# 将单词映射到整数索引的字典 
word_index = imdb.get_word_index()

# 第一个指数是保留的 
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
```

`Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
1646592/1641221 [==============================] - 0s 0us/step`                          
现在我们可以使用`decode_review`函数显示第一次检查的文本：

```
decode_review(train_data[0])
```

*"<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all"*

## 预处理数据

影评 – 整数数组，必须在输入神经网络之前转换为张量，这种转换可以通过以下两种方式完成：

* 将数组转换为0和1的向量，表示单词出现，类似于one-hot编码(热编码)。例如，序列[3,5]将成为10,000维向量，除了索引3和5（它们是1）之外全部为零。
然后，将其作为我们网络中的第一层(一个可以处理浮点矢量数据的Dense层)。但是，这种方法是内存密集型的，需要`num_words* num_reviews`大小矩阵。

* 或者，我们可以填充数组，使它们都具有相同的长度，然后创建一个整数张量的形状`max_length * num_reviews`。我们可以使用能够处理这种形状的嵌入层作为我们网络中的第一层。

在本教程中，我们将使用第二种方法。

由于电影评论的长度必须相同，我们将使用[pad_sequences](https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)函数来标准化长度：

```
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
```

我们再看一下数据的长度：

```
len(train_data[0]), len(train_data[1])
```
*(256, 256)*

并查看数据：

```
print(train_data[0])
```

*[ 1 14 22 16 43 530 973 1622 1385 65 458 4468 66 3941 4 173 36 256 5 25 100 43 838 112 50 670 2 9 35 480 284 5 150 4 172 112 167 2 336 385 39 4 172 4536 1111 17 546 38 13 447 4 192 50 16 6 147 2025 19 14 22 4 1920 4613 469 4 22 71 87 12 16 43 530 38 76 15 13 1247 4 22 17 515 17 12 16 626 18 2 5 62 386 12 8 316 8 106 5 4 2223 5244 16 480 66 3785 33 4 130 12 16 38 619 5 25 124 51 36 135 48 25 1415 33 6 22 12 215 28 77 52 5 14 407 16 82 2 8 4 107 117 5952 15 256 4 2 7 3766 5 723 36 71 43 530 476 26 400 317 46 7 4 2 1029 13 104 88 4 381 15 297 98 32 2071 56 26 141 6 194 7486 18 4 226 22 21 134 476 26 480 5 144 30 5535 18 51 36 28 224 92 25 104 4 226 65 16 38 1334 88 12 16 283 5 16 4472 113 103 32 15 16 5345 19 178 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]*

## 构建模型

神经网络是通过堆叠层创建的，这需要两个主要的架构决策：

* 模型中要使用多少层？
* 每层使用多少个隐藏单元？

在此示例中，输入数据由单词索引数组组成，要预测的标签是0或1，让我们为这个问题建立一个模型：

```
# 输入形状是用于电影评论的词汇计数（10,000字）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```                                                       

`Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________`

这些层按顺序堆叠以构建分类器：

1. 第一层是`Embedding`层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量。这些向量是作为模型训练学习的，向量为输入数组添加维度，生成的维度为：`(batch, sequence, embedding)`.

2. 接下来，`GlobalAveragePooling1D`层通过对序列维度求平均，为每个示例返回固定长度的输出向量。这允许模型以尽可能简单的方式处理可变长度。

3. 该固定长度输出适量通过具有16个隐藏单元的完全连接（`Dense`）层进行管道传输。

4. 最后一层与单个节点密集连接，使用`sigmoid`激活函数，此值介于0和1之间的浮点数，表示概率或置信度。

### 隐藏单元

上述模型在输入和输出之间有两个中间或“隐藏”层。输出的数量（单位，节点或神经元）是层的表示空间的维度。换句话说，在学习内部表示是允许网络的自由度。

如果模型具有更多隐藏单元（更高维度的表示空间）或更多层，则网络可以学习更复杂的表示。但是，它使网络的计算成本更高，并且可能导致学习不需要的模式，这些模式可以提高训练数据的性能，但不会提高测试数据的性能，这称为过度拟合*overfitting*，我们稍后会进行探讨。

### 损失函数和优化器

模型需要一个损失函数和一个用于训练的优化器。由于这是一个二元分类问题，并且模型输出概率（网络最后一层使用sigmoid 激活函数，仅包含一个单元），那么最好使用`binary_crossentropy`（二元交叉熵）损失。

这不是损失函数的唯一选择，例如，您可以选择`mean_squared_error`（均方误差）。但对于输出概率值的模型，交叉熵（crossentropy）往往是最好
的选择。交叉熵是来自于信息论领域的概念，用于衡量概率分布之间的距离，在这个例子中就是真实分布与预测值之间的距离。。

在后面，当我们探索回归问题（比如预测房子的价格）时，我们将看到如何使用另一种称为均方误差的损失函数。

现在，配置模型以使用优化器和损失函数：

```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## 创建验证集

在训练时，我们想要检查模型在以前没有见过的数据上的准确性。通过从原始训练数据中分离10,000个示例来创建验证集。（为什么不立即使用测试集？我们的目标是仅使用训练数据开发和调整我们的模型，然后仅使用测试数据来评估我们的准确性）。

```
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
```

## 训练模型

以512个样本的小批量训练模型40个周期，这是`x_train`和`y_train`张量中所有样本的40次迭代。在训练期间，监控模型在验证集中的10,000个样本的损失和准确性：

```
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```

`Epoch 40/40
15000/15000 [==============================] - 1s 54us/sample - loss: 0.0926 - accuracy: 0.9771 - val_loss: 0.3133 - val_accuracy: 0.8824`

## 评估模型

让我们看看模型的表现，将返回两个值，损失（表示我们的错误的数字，更低的值更好）和准确性。

```
results = model.evaluate(test_data, test_labels)

print(results)
```

`25000/25000 [==============================] - 1s 45us/sample - loss: 0.3334 - accuracy: 0.8704
[0.33341303256988525, 0.87036]`

这种相当简单的方法实现了约87％的准确度，使用更先进的方法，模型应该接近95％。

## 创建准确性和损失随时间变化的图表

`model.fit()`返回一个`History`对象，其中包含一个字典，其中包含训练期间发生的所有事情：

```
history_dict = history.history
history_dict.keys()
```

*dict_keys(['loss', 'val_loss', 'accuracy', 'val_accuracy'])*

有四个条目：在训练和验证期间，每个条目对应一个监控指标，我们可以使用这些来绘制训练和验证损失以进行比较，以及训练和验证准确性：

```
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

*<Figure size 640x480 with 1 Axes>*  

```
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```

![](https://tensorflow.google.cn/alpha/tutorials/keras/basic_text_classification_files/output_40_0.png)

在该图中，点表示训练损失和准确度，实线表示验证损失和准确度。

请注意，训练损失随着每个周期而*减少*，并且训练准确度随着每个周期而*增加*。当使用梯度下降优化时，这是预期的，它应该在每次迭代时最小化期望的数量。

这不是验证损失和准确性的情况，它们似乎在大约第20轮之后达到峰值。这是过度拟合的一个例子：模型在训练数据上的表现比在以前从未见过的数据上表现得更好。在此之后，模型过度优化并学习特定于训练数据的表示，这些表示不会推广到测试数据。

对于这种特殊情况，我们可以通过在第20轮左右停止训练来防止过度拟合。稍后，您将看到如何使用回调自动执行此操作。
