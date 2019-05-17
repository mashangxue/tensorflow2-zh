---
title: 使用RNN对文本进行分类
categories: tensorflow2官方文档
tags: tensorflow2.0
date: 2019-05-20
abbrlink: tensorflow/tensorflow2-tutorials-text-text_classification_rnn
---

# 使用RNN对文本进行分类

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/text/text_classification_rnn"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/text_classification_rnn.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/text_classification_rnn.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>


本文本分类教程在[IMDB大型影评数据集](http://ai.stanford.edu/~amaas/data/sentiment/) 上训练一个循环神经网络进行情感分类。

```python
from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow_datasets as tfds
import tensorflow as tf
```

导入matplotlib并创建一个辅助函数来绘制图形

```
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
```

## 设置输入管道


IMDB大型电影影评数据集是一个二元分类数据集，所有评论都有正面或负面的情绪标签。

使用[TFDS](https://tensorflow.google.cn/datasets)下载数据集，数据集附带一个内置的子字标记器


```python
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
```

由于这是一个子字标记器，它可以传递任何字符串，并且标记器将对其进行标记。

```python
tokenizer = info.features['text'].encoder

print ('Vocabulary size: {}'.format(tokenizer.vocab_size))
```
```
      Vocabulary size: 8185
```


```python
sample_string = 'TensorFlow is cool.'

tokenized_string = tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string
```

```
      Tokenized string is [6307, 2327, 4043, 4265, 9, 2724, 7975]
      The original string: TensorFlow is cool.
```

如果字符串不在字典中，则标记生成器通过将字符串分解为子字符串来对字符串进行编码。

```python
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))
```

```
    6307 ----> Ten
    2327 ----> sor
    4043 ----> Fl
    4265 ----> ow
    9 ----> is
    2724 ----> cool
    7975 ----> .
```


```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)
```

## 创建模型

构建一个`tf.keras.Sequential`模型并从嵌入层开始，嵌入层每个字存储一个向量，当被调用时，它将单词索引的序列转换为向量序列，这些向量是可训练的，在训练之后（在足够的数据上），具有相似含义的词通常具有相似的向量。

这种索引查找比通过`tf.keras.layers.Dense`层传递独热编码向量的等效操作更有效。

递归神经网络（RNN）通过迭代元素来处理序列输入，RNN将输出从一个时间步传递到其输入端，然后传递到下一个时间步。

`tf.keras.layers.Bidirectional`包装器也可以与RNN层一起使用。这通过RNN层向前和向后传播输入，然后连接输出。这有助于RNN学习远程依赖性。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译Keras模型以配置训练过程：
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

## 训练模型

```python
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)
```

```
      ...
      Epoch 10/10
      391/391 [==============================] - 70s 180ms/step - loss: 0.3074 - accuracy: 0.8692 - val_loss: 0.5533 - val_accuracy: 0.7873
```


```python
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
```

```
          391/Unknown - 19s 47ms/step - loss: 0.5533 - accuracy: 0.7873Test Loss: 0.553319326714
      Test Accuracy: 0.787320017815
```


上面的模型没有屏蔽应用于序列的填充。如果我们对填充序列进行训练，并对未填充序列进行测试，就会导致偏斜。理想情况下，模型应该学会忽略填充，但是正如您在下面看到的，它对输出的影响确实很小。

如果预测 >=0.5，则为正，否则为负。

```python
def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sentence, pad):
  tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)

  if pad:
    tokenized_sample_pred_text = pad_to_size(tokenized_sample_pred_text, 64)

  predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

  return (predictions)
```


```python
# 对不带填充的示例文本进行预测 

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)
```

```
        [[ 0.68914342]]
```


```python
# 对带填充的示例文本进行预测 

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)
```

```
       [[ 0.68634349]]
```

```python
plot_graphs(history, 'accuracy')
```

![png](https://tensorflow.google.cn/alpha/tutorials/sequences/text_classification_rnn_files/output_29_0.png)


```python
plot_graphs(history, 'loss')
```

![png](https://tensorflow.google.cn/alpha/tutorials/sequences/text_classification_rnn_files/output_30_0.png)


## 堆叠两个或更多LSTM层


Keras递归层有两种可以用的模式，由`return_sequences`构造函数参数控制：

* 返回每个时间步的连续输出的完整序列（3D张量形状 `(batch_size, timesteps, output_features)`）。

* 仅返回每个输入序列的最后一个输出（2D张量形状 `(batch_size, output_features)`）。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)
```

```
      ...
      Epoch 10/10
      391/391 [==============================] - 154s 394ms/step - loss: 0.1120 - accuracy: 0.9643 - val_loss: 0.5646 - val_accuracy: 0.8070
```

```python
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
```

```
            391/Unknown - 45s 115ms/step - loss: 0.5646 - accuracy: 0.8070Test Loss: 0.564571284348
        Test Accuracy: 0.80703997612
```


```python
# 在没有填充的情况下预测示例文本

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)
```

```
       [[ 0.00393916]]
```


```python
# 在有填充的情况下预测示例文本

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)
```

```
      [[ 0.01098633]]
```


```python
plot_graphs(history, 'accuracy')
```

![png](https://tensorflow.google.cn/alpha/tutorials/sequences/text_classification_rnn_files/output_38_0.png)



```python
plot_graphs(history, 'loss')
```

![png](https://tensorflow.google.cn/alpha/tutorials/sequences/text_classification_rnn_files/output_39_0.png)

查看其它现有的递归层，例如[GRU层](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRU)。

