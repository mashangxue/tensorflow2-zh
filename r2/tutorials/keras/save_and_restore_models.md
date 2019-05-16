---
title: 保存和加载模型
categories: tensorflow2.0官方文档
tags: tensorflow2.0
date: 2019-05-20
abbrlink: tensorflow/tensorflow2-tutorials-keras-save_and_restore_models
---

# 保存和加载模型

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

Caution: Be careful with untrusted code—TensorFlow models are code. See [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) for details.

模型进度可以在训练期间和训练后保存。这意味着模型可以在它停止的地方继续，并避免长时间的训练。保存还意味着您可以共享您的模型，其他人可以重新创建您的工作。当发布研究模型和技术时，大多数机器学习实践者共享:
* 用于创建模型的代码
* 以及模型的训练权重或参数

共享此数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。

注意：小心不受信任的代码(TensorFlow模型是代码)。有关详细信息，请参阅[安全使用TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) 。

**选项**

保存TensorFlow模型有多种方法，具体取决于你使用的API。本章节使用tf.keras(一个高级API，用于TensorFlow中构建和训练模型)，有关其他方法，请参阅TensorFlow[保存和还原指南](https://tensorflow.google.cn/guide/saved_model)或[保存在eager中](https://tensorflow.google.cn/guide/eager#object-based_saving)。

## 1. 设置

### 1.1. 安装和导入

需要安装和导入TensorFlow和依赖项

```
pip install h5py pyyaml
```

### 1.2. 获取样本数据集

我们将使用[MNIST数据集](http://yann.lecun.com/exdb/mnist/)来训练我们的模型以演示保存权重，要加速这些演示运行，请只使用前1000个样本数据：

```
from __future__ import absolute_import, division, print_function, unicode_literals

import os

!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf
from tensorflow import keras

tf.__version__
```


```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

### 1.3. 定义模型

让我们构建一个简单的模型，我们将用它来演示保存和加载权重。

```
# 返回一个简短的序列模型 
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model


# 创建基本模型实例
model = create_model()
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

## 2. 在训练期间保存检查点

The primary use case is to automatically save checkpoints *during* and at *the end* of training. This way you can use a trained model without having to retrain it, or pick-up training where you left of—in case the training process was interrupted.

`tf.keras.callbacks.ModelCheckpoint` is a callback that performs this task. The callback takes a couple of arguments to configure checkpointing.

主要用例是在训练期间和训练结束时自动保存检查点，通过这种方式，您可以使用训练有素的模型，而无需重新训练，或者在您离开的地方继续训练，以防止训练过程中断。

`tf.keras.callbacks.ModelCheckpoint`是执行此任务的回调，回调需要几个参数来配置检查点。

### 2.1. 检查点回调使用情况

训练模型并将其传递给 `ModelCheckpoint`回调

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）是为了阻止过时使用的，可以忽略。
```

```
Train on 1000 samples, validate on 1000 samples
......
Epoch 10/10
 960/1000 [===========================>..] - ETA: 0s - loss: 0.0392 - accuracy: 1.0000
Epoch 00010: saving model to training_1/cp.ckpt
1000/1000 [==============================] - 0s 207us/sample - loss: 0.0393 - accuracy: 1.0000 - val_loss: 0.3976 - val_accuracy: 0.8750

<tensorflow.python.keras.callbacks.History at 0x7efc3eba7358>
```

这将创建一个TensorFlow检查点文件集合，这些文件在每个周期结束时更新。
文件夹checkpoint_dir下的内容如下：（Linux系统使用 `ls`命令查看）
```
checkpoint  cp.ckpt.data-00000-of-00001  cp.ckpt.index
```

创建一个新的未经训练的模型，仅从权重恢复模型时，必须具有与原始模型具有相同体系结构的模型，由于它是相同的模型架构，我们可以共享权重，尽管它是模型的不同示例。

现在重建一个新的，未经训练的模型，并在测试集中评估它。未经训练的模型将在随机水平(约10%的准确率):

```
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
```

```
1000/1000 [==============================] - 0s 107us/sample - loss: 2.3224 - accuracy: 0.1230
Untrained model, accuracy: 12.30%
```

然后从检查点加载权重，并重新评估：

```
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

```
1000/1000 [==============================] - 0s 48us/sample - loss: 0.3976 - accuracy: 0.8750
Restored model, accuracy: 87.50%
```

### 2.2. 检查点选项

回调提供了几个选项，可以为生成的检查点提供唯一的名称，并调整检查点频率。

训练一个新模型，每5个周期保存一次唯一命名的检查点：

```
# 在文件名中包含周期数. (使用 `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 每5个周期保存一次权重
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
```

```

Epoch 00005: saving model to training_2/cp-0005.ckpt
......
Epoch 00050: saving model to training_2/cp-0050.ckpt
<tensorflow.python.keras.callbacks.History at 0x7efc7c3bbd30>
```

现在，查看生成的检查点并选择最新的检查点：

```
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
```

'''
'training_2/cp-0050.ckpt'
'''

注意：默认的tensorflow格式仅保存最近的5个检查点。

要测试，请重置模型并加载最新的检查点：

```
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

```
1000/1000 [==============================] - 0s 84us/sample - loss: 0.4695 - accuracy: 0.8810
Restored model, accuracy: 88.10%
```

## 3. 这些文件是什么？

上述代码将权重存储到[检查点]((https://tensorflow.google.cn/guide/saved_model#save_and_restore_variables))格式的文件集合中，这些文件仅包含二进制格式的训练权重.
检查点包含：
* 一个或多个包含模型权重的分片；
* 索引文件，指示哪些权重存储在哪个分片。

如果您只在一台机器上训练模型，那么您将有一个带有后缀的分片：`.data-00000-of-00001`


## 4. 手动保存权重

上面你看到了如何将权重加载到模型中。手动保存权重同样简单，使用`Model.save_weights`方法。

```
# 保存权重
model.save_weights('./checkpoints/my_checkpoint')

# 加载权重
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

## 5. 保存整个模型

模型和优化器可以保存到包含其状态（权重和变量）和模型配置的文件中，这允许您导出模型，以便可以在不访问原始python代码的情况下使用它。由于恢复了优化器状态，您甚至可以从中断的位置恢复训练。

保存完整的模型非常有用，您可以在TensorFlow.js([HDF5](https://tensorflow.google.cn/js/tutorials/import-keras.html), [Saved Model](https://tensorflow.google.cn/js/tutorials/conversion/import_saved_model)) 中加载它们，然后在Web浏览器中训练和运行它们，或者使用TensorFlow Lite([HDF5](https://tensorflow.google.cn/lite/convert/python_api#exporting_a_tfkeras_file_), [Saved Model](https://tensorflow.google.cn/lite/convert/python_api#exporting_a_savedmodel_))将它们转换为在移动设备上运行。

### 5.1. 作为HDF5文件

Keras使用[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)标准提供基本保存格式，出于我们的目的，可以将保存的模型视为单个二进制blob。


```
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# 保存整个模型到HDF5文件 
model.save('my_model.h5')
```

现在从该文件重新创建模型：

```
# 重新创建完全相同的模型，包括权重和优化器
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
```

```
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_12 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_13 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

检查模型的准确率:


```
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

```
1000/1000 [==============================] - 0s 94us/sample - loss: 0.4137 - accuracy: 0.8540
Restored model, accuracy: 85.40%
```

此方法可保存模型的所有东西：
* 权重值
* 模型的配置（架构）
* 优化器配置

Keras通过检查架构来保存模型，目前它无法保存TensorFlow优化器（来自`tf.train`）。使用这些时，您需要在加载后重新编译模型，否则您将失去优化程序的状态。


### 5.2. 作为 `saved_model`

注意：这种保存`tf.keras`模型的方法是实验性的，在将来的版本中可能会有所改变。

创建一个新的模型：

```
model = create_model()

model.fit(train_images, train_labels, epochs=5)
```

创建`saved_model`，并将其放在带时间戳的目录中：

```
import time
saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path
```

```
...
'./saved_models/1555630614'
```

从保存的模型重新加载新的keras模型：

```
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()
```

```
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_14 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_7 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

运行加载的模型进行预测：

```
model.predict(test_images).shape
```

```
(1000, 10)
```


```
# 必须要在评估之前编译模型
# 如果仅部署已保存的模型，则不需要此步骤 

new_model.compile(optimizer=model.optimizer,  # keep the optimizer that was loaded
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 评估加载后的模型 
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

```
1000/1000 [==============================] - 0s 102us/sample - loss: 0.4367 - accuracy: 0.8570
Restored model, accuracy: 85.70%
```

## 6. 下一步是什么

这是使用`tf.keras`保存和加载的快速指南。

* [tf.keras指南](https://tensorflow.google.cn/guide/keras)显示了有关使用tf.keras保存和加载模型的更多信息。

* 在eager execution期间保存，请参阅在[Saving in eager](https://tensorflow.google.cn/guide/eager#object_based_saving)。

* [保存和还原指南](https://tensorflow.google.cn/guide/saved_model)包含有关TensorFlow保存的低阶详细信息。

