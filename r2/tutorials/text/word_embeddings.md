

# 使用词嵌入

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/text/word_embeddings.ipynb">
    <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
    View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/word_embeddings.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/word_embeddings.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

本章节介绍了词嵌入，它包含完整的代码，可以在小型数据集上从零开始训练词嵌入，并使用嵌入投影仪[Embedding Projector](http://projector.tensorflow.org) 可视化这些嵌入，如下图所示：

<img src="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/images/embedding.jpg?raw=1" alt="Screenshot of the embedding projector" width="400"/>

## 将文本表示为数字

机器学习模型以向量（数字数组）作为输入，在处理文本时，我们必须首先想出一个策略，将字符串转换为数字（或将文本“向量化”），然后再将其提供给模型。在本节中，我们将研究三种策略。

### 独热编码（One-hot encodings）

首先，我们可以用“one-hot”对词汇的每个单词进行编码，想想“the cat sat on the mat”这句话，这个句子中的词汇（或独特的单词）是（cat,mat,on,The），为了表示每个单词，我们将创建一个长度等于词汇表的零向量，然后再对应单词的索引中放置一个1。这种方法如下图所示：

<img src="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/r2/tutorials/text/images/one-hot.png" alt="Diagram of one-hot encodings" width="400" />

为了创建包含句子编码的向量，我们可以连接每个单词的one-hot向量。

关键点：这种方法是低效的，一个热编码的向量是稀疏的（意思是，大多数指标是零）。假设我们有10000个单词，要对每个单词进行一个热编码，我们将创建一个向量，其中99.99%的元素为零。

### 用唯一的数字编码每个单词

我们尝试第二种方法，使用唯一的数字编码每个单词。继续上面的例子，我们可以将1赋值给“cat”，将2赋值给“mat”，以此类推，然后我们可以将句子“The cat sat on the mat”编码为像[5, 1, 4, 3, 5, 2]这样的密集向量。这种方法很有效，我们现有有一个稠密的向量（所有元素都是满的），而不是稀疏的向量。

然而，这种方法有两个缺点：

* 整数编码是任意的（它不捕获单词之间的任何关系）。

* 对于模型来说，整数编码的解释是很有挑战性的。例如，线性分类器为每个特征学习单个权重。由于任何两个单词的相似性与它们编码的相似性之间没有关系，所以这种特征权重组合没有意义。


### 词嵌入

词嵌入为我们提供了一种使用高效、密集表示的方法，其中相似的单词具有相似的编码，重要的是，我们不必手工指定这种编码，嵌入是浮点值的密集向量（向量的长度是您指定的参数），它们不是手工指定嵌入的值，而是可训练的参数（模型在训练期间学习的权重，与模型学习密集层的权重的方法相同）。通常会看到8维（对于小数据集）的词嵌入，在处理大型数据集时最多可达1024维。更高维度的嵌入可以捕获单词之间的细粒度关系，但需要更多的数据来学习。

![Diagram of an embedding](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/images/embedding2.png?raw=1)

上面是词嵌入的图表，每个单词表示为浮点值的4维向量，另一种考虑嵌入的方法是“查找表”，在学习了这些权重之后，我们可以通过查找表中对应的密集向量来编码每个单词。

## 使用嵌入层Embedding Layer

Keras makes it easy to use word embeddings. Let's take a look at the [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) layer.


```
from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tf-nightly-2.0-preview
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# The Embedding layer takes at least two arguments:
# the number of possible words in the vocabulary, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 32.
embedding_layer = layers.Embedding(1000, 32)
```

The Embedding layer can be understood as a lookup table that maps from integer indices (which stand for specific words) to dense vectors (their embeddings). The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem, much in the same way you would experiment with the number of neurons in a Dense layer.

When you create an Embedding layer, the weights for the embedding are randomly initialized (just like any other layer). During training, they are gradually adjusted via backpropagation. Once trained, the learned word embeddings will roughly encode similarities between words (as they were learned for the specific problem your model is trained on).

As input, the Embedding layer takes a 2D tensor of integers, of shape `(samples, sequence_length)`, where each entry is a sequence of integers. It can embed sequences of variable lengths. You could feed into the embedding layer above batches with shapes `(32, 10)` (batch of 32 sequences of length 10) or `(64, 15)` (batch of 64 sequences of length 15). All sequences in a batch must have the same length, so sequences that are shorter than others should be padded with zeros, and sequences that are longer should be truncated.

As output, the embedding layer returns a 3D floating point tensor, of shape `(samples, sequence_length, embedding_dimensionality)`. Such a 3D tensor can then be processed by a RNN layer, or can simply be flattened or pooled and processed by a Dense layer. We will show the latter approach in this tutorial, and you can refer to the [Text Classification with an RNN](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/text_classification_rnn.ipynb) to learn the first.

## Learning embeddings from scratch

We will train a sentiment classifier on IMDB movie reviews. In the process, we will learn embeddings from scratch. We will move quickly through the code that downloads and preprocesses the dataset (see this [tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification) for more details).


```
vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
```

As imported, the text of reviews is integer-encoded (each integer represents a specific word in a dictionary).


```
print(train_data[0])
```

### Convert the integers back to words

It may be useful to know how to convert integers back to text. Here, we'll create a helper function to query a dictionary object that contains the integer to string mapping:


```
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])
```

Movie reviews can be different lengths. We will use the `pad_sequences` function to standardize the lengths of the reviews.


```
maxlen = 500

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=maxlen)
```

Let's inspect the first padded review.


```
print(train_data[0])
```

### Create a simple model

We will use the [Keras Sequential API](https://www.tensorflow.org/guide/keras) to define our model.

* The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: `(batch, sequence, embedding)``.

* Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.

* This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

* The last layer is densely connected with a single output node. Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability (or confidence level) that the review is positive.


```
embedding_dim=16

model = keras.Sequential([
  layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()
```

### Compile and train the model


```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)
```

With this approach our model reaches a validation accuracy of around 88% (note the model is overfitting, training accuracy is significantly higher).


```
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))

plt.show()
```

## Retrieve the learned embeddings

Next, let's retrieve the word embeddings learned during training. This will be a matrix of shape `(vocab_size,embedding-dimension)`.


```
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
```

We will now write the weights to disk. To use the [Embedding Projector](http://projector.tensorflow.org), we will upload two files in tab separated format: a file of vectors (containing the embedding), and a file of meta data (containing the words).


```
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```

If you are running this tutorial in [Colaboratory](https://colab.research.google.com), you can use the following snippet to download these files to your local machine (or use the file browser, *View -> Table of contents -> File browser*).


```
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
```

## Visualize the embeddings

To visualize our embeddings we will upload them to the embedding projector.

Open the [Embedding Projector](http://projector.tensorflow.org/).

* Click on "Load data".

* Upload the two files we created above: ```vecs.tsv``` and ```meta.tsv```. T

The embeddings you have trained will now be displayed. You can search for words to find their closest neighbors. For example, try searching for "beautiful". You may see neighbors like "wonderful". Note: your results may be a bit different, depending on how weights were randomly initialized before training the embedding layer.

Note: experimentally, you may be able to produce more interpretable embeddings by using a simpler model. Try deleting the `Dense(16)` layer, retraining the model, and visualizing the embeddings again.

<img src="https://github.com/tensorflow/docs/blob/88d3b244eca8088c6c406ff9bdc8505d7fb6cbe7/site/en/r2/tutorials/text/images/embedding.jpg?raw=1" alt="Screenshot of the embedding projector" width="400"/>


## Next steps


This tutorial has shown you how to train and visualize word embeddings from scratch on a small dataset.

* To learn more about embeddings in Keras we recommend these [notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.2-understanding-recurrent-neural-networks.ipynb) by François Chollet.

* To learn more about text classification (including the overall workflow, and if you're curious about when to use embeddings vs one-hot encodings) we recommend this practical text classification [guide](https://developers.google.com/machine-learning/guides/text-classification/step-2-5).
