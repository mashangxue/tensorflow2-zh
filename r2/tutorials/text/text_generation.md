---
title: 使用RNN生成文本实战：莎士比亚风格诗句  (tensorflow2.0官方教程翻译）
categories: tensorflow2官方教程
tags: tensorflow2.0教程
top: 1961
abbrlink: tensorflow/tf2-tutorials-text-text_generation
---

# 使用RNN生成文本实战：莎士比亚风格诗句  (tensorflow2.0官方教程翻译）

本教程演示了如何使用基于字符的 RNN 生成文本。我们将使用 Andrej Karpathy 在 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 一文中提供的莎士比亚作品数据集。我们根据此数据（“Shakespear”）中的给定字符序列训练一个模型，让它预测序列的下一个字符（“e”）。通过重复调用该模型，可以生成更长的文本序列。


注意：启用 GPU 加速可提高执行速度。在 Colab 中依次选择“运行时”>“更改运行时类型”>“硬件加速器”>“GPU”。如果在本地运行，请确保 TensorFlow 的版本为 1.11.0 或更高版本。

本教程中包含使用 [tf.keras](https://tensorflow.google.cn/guide/keras) 和 [Eager Execution](https://tensorflow.google.cn/guide/eager) 实现的可运行代码。以下是本教程中的模型训练了30个周期时的示例输出，并以字符串“Q”开头：

<pre>
QUEENE:
I had thought thou hadst a Roman; for the oracle,
Thus by All bids the man against the word,
Which are so weak of care, by old care done;
Your children were in your holy love,
And the precipitation through the bleeding throne.

BISHOP OF ELY:
Marry, and will, my lord, to weep in such a one were prettiest;
Yet now I was adopted heir
Of the world's lamentable day,
To watch the next way with his father with his face?

ESCALUS:
The cause why then we are all resolved more sons.

VOLUMNIA:
O, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, it is no sin it should be dead,
And love and pale as any will to that word.

QUEEN ELIZABETH:
But how long have I heard the soul for this world,
And show his hands of life be proved to stand.

PETRUCHIO:
I say he look'd on, if I must be content
To stay him from the fatal of our country's bliss.
His lordship pluck'd from this sentence then for prey,
And then let us twain, being the moon,
were she such a case as fills m
</pre>

虽然有些句子合乎语法规则，但大多数句子都没有意义。该模型尚未学习单词的含义，但请考虑以下几点：

* 该模型是基于字符的模型。在训练之初，该模型都不知道如何拼写英语单词，甚至不知道单词是一种文本单位。

* 输出的文本结构仿照了剧本的结构：文本块通常以讲话者的名字开头，并且像数据集中一样，这些名字全部采用大写字母。

* 如下文所示，尽管该模型只使用小批次的文本（每批文本包含 100 个字符）训练而成，但它仍然能够生成具有连贯结构的更长文本序列。

## 1. 设置Setup

### 1.1. 导入 TensorFlow 和其他库


```python
from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import numpy as np
import os
import time
```

```
    Collecting tensorflow-gpu==2.0.0-alpha0
    Successfully installed google-pasta-0.1.4 tb-nightly-1.14.0a20190303 tensorflow-estimator-2.0-preview-1.14.0.dev2019030300 tensorflow-gpu==2.0.0-alpha0-2.0.0.dev20190303
```

### 1.2. 下载莎士比亚数据集


通过更改以下行可使用您自己的数据运行此代码。

```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

### 1.3. 读取数据

首先，我们来看一下文本内容。

```python
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))
```

    Length of text: 1115394 characters



```python
# Take a look at the first 250 characters in text
print(text[:250])
```

    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    



```python
# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
```

    65 unique characters


## 2. 处理文本

### 2.1. 向量化文本

在训练之前，我们需要将字符串映射到数字表示值。创建两个对照表：一个用于将字符映射到数字，另一个用于将数字映射到字符。

```python
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
```

现在，每个字符都有一个对应的整数表示值。请注意，我们按从 0 到 `len(unique)` 的索引映射字符。

```python
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
```

    {
      '\n':   0,
      ' ' :   1,
      '!' :   2,
      ...
      'F' :  18,
      'G' :  19,
      ...
    }



```
# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
```

    'First Citizen' ---- characters mapped to int ---- > [18 47 56 57 58  1 15 47 58 47 64 43 52]


### 2.2. 预测任务

根据给定的字符或字符序列预测下一个字符最有可能是什么？这是我们要训练模型去执行的任务。模型的输入将是字符序列，而我们要训练模型去预测输出，即每一个时间步的下一个字符。

由于 RNN 会依赖之前看到的元素来维持内部状态，那么根据目前为止已计算过的所有字符，下一个字符是什么？

### 2.3. 创建训练样本和目标

将文本划分为训练样本和训练目标。每个训练样本都包含从文本中选取的 `seq_length` 个字符。

相应的目标也包含相同长度的文本，但是将所选的字符序列向右顺移一个字符。

将文本拆分成文本块，每个块的长度为 `seq_length+1` 个字符。例如，假设 `seq_length` 为 4，我们的文本为“Hello”，则可以将“Hell”创建为训练样本，将“ello”创建为目标。

为此，首先使用`tf.data.Dataset.from_tensor_slices`函数将文本向量转换为字符索引流。

```python
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
```

    F
    i
    r
    s
    t


批处理方法可以让我们轻松地将这些单个字符转换为所需大小的序列。

```python
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))
```

    'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '
    'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'
    "now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"
    "ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"
    'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'


对于每个序列，复制并移动它以创建输入文本和目标文本，方法是使用 `map` 方法将简单函数应用于每个批处理：

```python
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
```

打印第一个样本输入和目标值：

```python
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
```

    Input data:  'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
    Target data: 'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '

这些向量的每个索引均作为一个时间步来处理。对于时间步 0 的输入，我们收到了映射到字符 “F” 的索引，并尝试预测 “i” 的索引作为下一个字符。在下一个时间步，执行相同的操作，但除了当前字符外，`RNN` 还要考虑上一步的信息。

```python
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
```

    Step    0
      input: 18 ('F')
      expected output: 47 ('i')
    ...
    Step    4
      input: 58 ('t')
      expected output: 1 (' ')


### 2.4. 使用 tf.data 创建批次文本并重排这些批次

我们使用 `tf.data` 将文本拆分为可管理的序列。但在将这些数据馈送到模型中之前，我们需要对数据进行重排，并将其打包成批。

```
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
```

```
    <BatchDataset shapes: ((64, 100), (64, 100)), types: (tf.int64, tf.int64)>
```

## 3. 实现模型

使用`tf.keras.Sequential`来定义模型。对于这个简单的例子，我们可以使用三个层来定义模型：

* `tf.keras.layers.Embedding`：嵌入层（输入层）。一个可训练的对照表，它会将每个字符的数字映射到具有 `embedding_dim` 个维度的高维度向量；

* `tf.keras.layers.GRU`：  GRU 层：一种层大小等于单位数(`units = rnn_units`)的 RNN。（在此示例中，您也可以使用 LSTM 层。）

* `tf.keras.layers.Dense`：密集层（输出层），带有`vocab_size`个单元输出。

```python
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
```


```python
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
```


```python
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
```

对于每个字符，模型查找嵌入，以嵌入作为输入一次运行GRU，并应用密集层生成预测下一个字符的对数可能性的logits：

![A drawing of the data passing through the model](https://raw.githubusercontent.com/mari-linhares/docs/patch-1/site/en/tutorials/sequences/images/text_generation_training.png)

## 4. 试试这个模型

现在运行模型以查看它的行为符合预期，首先检查输出的形状：


```python
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
```

    (64, 100, 65) # (batch_size, sequence_length, vocab_size)


在上面的示例中，输入的序列长度为 `100` ，但模型可以在任何长度的输入上运行：

```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding (Embedding)        (64, None, 256)           16640
    _________________________________________________________________
    unified_lstm (UnifiedLSTM)   (64, None, 1024)          5246976
    _________________________________________________________________
    dense (Dense)                (64, None, 65)            66625
    =================================================================
    Total params: 5,330,241
    Trainable params: 5,330,241
    Non-trainable params: 0
    _________________________________________________________________


为了从模型中获得实际预测，我们需要从输出分布中进行采样，以获得实际的字符索引。此分布由字符词汇表上的logits定义。

注意：从这个分布中进行_sample_（采样）非常重要，因为获取分布的_argmax_可以轻松地将模型卡在循环中。

尝试批处理中的第一个样本：

```python
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
```

这使我们在每个时间步都预测下一个字符索引：

```python
sampled_indices
```


    array([21,  2, 58, 40, 42, 32, 39,  7, 18, 38, 30, 58, 23, 58, 37, 10, 23,
           16, 52, 14, 43,  8, 32, 49, 62, 41, 53, 38, 17, 36, 24, 59, 41, 38,
            4, 27, 33, 59, 54, 34, 14,  1,  1, 56, 55, 40, 37,  4, 32, 44, 62,
           59,  1, 10, 20, 29,  2, 48, 37, 26, 10, 22, 58,  5, 26,  9, 23, 26,
           54, 43, 46, 36, 62, 57,  8, 53, 52, 23, 57, 42, 60, 10, 43, 11, 45,
           12, 28, 46, 46, 15, 51,  9, 56,  7, 53, 51,  2,  1, 10, 58])


解码这些以查看此未经训练的模型预测的文本：


```python
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
```

    Input:
     'to it far before thy time?\nWarwick is chancellor and the lord of Calais;\nStern Falconbridge commands'
    
    Next Char Predictions:
     "I!tbdTa-FZRtKtY:KDnBe.TkxcoZEXLucZ&OUupVB  rqbY&Tfxu :HQ!jYN:Jt'N3KNpehXxs.onKsdv:e;g?PhhCm3r-om! :t"


## 5. 训练模型

此时，问题可以被视为标准分类问题。给定先前的RNN状态，以及此时间步的输入，预测下一个字符的类。

### 5.1. 添加优化器和损失函数

标准的`tf.keras.losses.sparse_softmax_crossentropy`损失函数在这种情况下有效，因为它应用于预测的最后一个维度。

因为我们的模型返回logits，所以我们需要设置`from_logits`标志。

```python
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
```

    Prediction shape:  (64, 100, 65)  # (batch_size, sequence_length, vocab_size)
    scalar_loss:       4.174188

使用 `tf.keras.Model.compile` 方法配置培训过程。我们将使用带有默认参数和损失函数的 `tf.keras.optimizers.Adam`。


```python
model.compile(optimizer='adam', loss=loss)
```

### 5.2. 配置检查点 

使用`tf.keras.callbacks.ModelCheckpoint`确保在训练期间保存检查点：

```python
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
```

### 5.3. 开始训练

为了使训练时间合理，使用10个时期来训练模型。在Colab中，将运行时设置为GPU以便更快地进行训练。

```python
EPOCHS=10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
```

    Epoch 1/10
    172/172 [==============================] - 31s 183ms/step - loss: 2.7052
    ......
    Epoch 10/10
    172/172 [==============================] - 31s 180ms/step - loss: 1.2276


## 6. 生成文本

### 6.1. 加载最新的检查点

要使此预测步骤简单，请使用批处理大小1。

由于RNN状态从时间步长传递到时间步的方式，模型一旦构建就只接受固定大小的批次数据。

要使用不同的 `batch_size` 运行模型，我们需要重建模型并从检查点恢复权重。

```python
tf.train.latest_checkpoint(checkpoint_dir)
```

```
        './training_checkpoints/ckpt_10'
```

```python
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_1 (Embedding)      (1, None, 256)            16640
    _________________________________________________________________
    unified_lstm_1 (UnifiedLSTM) (1, None, 1024)           5246976
    _________________________________________________________________
    dense_1 (Dense)              (1, None, 65)             66625
    =================================================================
    Total params: 5,330,241
    Trainable params: 5,330,241
    Non-trainable params: 0
    _________________________________________________________________


### 6.2. 预测循环

下面的代码块可生成文本：

* 首先选择一个起始字符串，初始化 RNN 状态，并设置要生成的字符数。

* 使用起始字符串和 RNN 状态获取预测值。

* 然后，使用多项分布计算预测字符的索引。 将此预测字符用作模型的下一个输入。

* 模型返回的 RNN 状态被馈送回模型中，使模型现在拥有更多上下文，而不是仅有一个单词。在模型预测下一个单词之后，经过修改的 RNN 状态再次被馈送回模型中，模型从先前预测的单词获取更多上下文，从而通过这种方式进行学习。


![To generate text the model's output is fed back to the input](https://github.com/mari-linhares/docs/blob/patch-1/site/en/tutorials/sequences/images/text_generation_sampling.png?raw=true)

查看生成的文本后，您会发现模型知道何时应使用大写字母，以及如何构成段落和模仿莎士比亚风格的词汇。由于执行的训练周期较少，因此该模型尚未学会生成连贯的句子。

```python
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```


```python
print(generate_text(model, start_string=u"ROMEO: "))
```

    ROMEO: now to have weth hearten sonce,
    No more than the thing stand perfect your self,
    Love way come. Up, this is d so do in friends:
    If I fear e this, I poisple
    My gracious lusty, born once for readyus disguised:
    But that a pry; do it sure, thou wert love his cause;
    My mind is come too!
    
    POMPEY:
    Serve my master's him: he hath extreme over his hand in the
    where they shall not hear they right for me.
    
    PROSSPOLUCETER:
    I pray you, mistress, I shall be construted
    With one that you shall that we know it, in this gentleasing earls of daiberkers now
    he is to look upon this face, which leadens from his master as
    you should not put what you perciploce backzat of cast,
    Nor fear it sometime but for a pit
    a world of Hantua?
    
    First Gentleman:
    That we can fall of bastards my sperial;
    O, she Go seeming that which I have
    what enby oar own best injuring them,
    Or thom I do now, I, in heart is nothing gone,
    Leatt the bark which was done born.
    
    BRUTUS:
    Both Margaret, he is sword of the house person. If born,


如果要改进结果，最简单的方法是增加模型训练的时长（请尝试 EPOCHS=30）。

您还可以尝试使用不同的起始字符，或尝试添加另一个 RNN 层以提高模型的准确率，又或者调整温度参数以生成具有一定随机性的预测值。

## 7. 高级：自定义训练

上述训练程序很简单，但不会给你太多控制。

所以现在您已经了解了如何手动运行模型，让我们解压缩训练循环，并自己实现。例如，如果要实施课程学习以帮助稳定模型的开环输出，这就是一个起点。

We will use `tf.GradientTape` to track the gradiends. You can learn more about this approach by reading the [eager execution guide](https://www.tensorflow.org/guide/eager).

我们将使用 `tf.GradientTape` 来跟踪梯度。您可以通过阅读[eager execution guide](https://www.tensorflow.org/guide/eager)来了解有关此方法的更多信息。

该程序的工作原理如下：

* 首先，初始化 RNN 状态。 我们通过调用 `tf.keras.Model.reset_states` 方法来完成此操作。
* 接下来，迭代数据集（逐批）并计算与每个数据集关联的预测。
* 打开 `tf.GradientTape` ，计算该上下文中的预测和损失。
* 使用 `tf.GradientTape.grads` 方法计算相对于模型变量的损失梯度。
* 最后，使用优化器的 `tf.train.Optimizer.apply_gradients` 方法向下迈出一步。

```python
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()
```


```python
@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss
```


```python
# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # saving (checkpoint) the model every 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
```

```
    .....
    Epoch 10 Batch 0 Loss 1.2350478172302246
    Epoch 10 Batch 100 Loss 1.1610674858093262
    Epoch 10 Loss 1.1558
    Time taken for 1 epoch 14.261839628219604 sec
```    

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-tutorials-text-text_generation.html](https://www.mashangxue123.com/tensorflow/tf2-tutorials-text-text_generation.html)
> 英文版本：[https://tensorflow.google.cn/alpha/tutorials/text/text_generation](https://tensorflow.google.cn/alpha/tutorials/text/text_generation)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/text/text_generation.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/text/text_generation.md)
