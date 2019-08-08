---
title: 图像分割
tags: tensorflow2.0教程
categories: tensorflow2官方教程
top: 1924
abbrlink: tensorflow/tf2-tutorials-images-intro_to_cnns
---

# 图像分割 (tensorflow2.0官方教程翻译)

> 最新版本：[https://www.mashangxue123.com/tensorflow/tf2-tutorials-images-segmentation.html](https://www.mashangxue123.com/tensorflow/tf2-tutorials-images-segmentation.html)
> 英文版本：[https://tensorflow.google.cn/beta/tutorials/images/segmentation](https://tensorflow.google.cn/beta/tutorials/images/segmentation)
> 翻译建议PR：[https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/images/segmentation.md](https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/images/segmentation.md)


本教程重点介绍使用修改后的[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)进行图像分割的任务。

## 什么是图像分割？

前面的章节我们学习了图像分类，网络算法的任务是为输入图像输出对应的标签或类。但是，假设您想知道对象在图像中的位置，该对象的形状，哪个像素属于哪个对象等。在这种情况下，您将要分割图像，即图像的每个像素都是给了一个标签。

因此，图像分割的任务是训练神经网络以输出图像的逐像素掩模。这有助于以更低的水平（即像素级别）理解图像。图像分割在医学成像，自动驾驶汽车和卫星成像等方面具有许多应用。

将用于本教程的数据集是由Parkhi等人创建的[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)。数据集由图像、其对应的标签和像素方式的掩码组成。掩模基本上是每个像素的标签。每个像素分为三类：
*   第1类：属于宠物的像素。
*   第2类：与宠物接壤的像素。
*   第3类：以上都没有/周围像素。

下载依赖项目  https://github.com/tensorflow/examples，
把文件夹tensorflow_examples放到项目下，下面会导入pix2pix

安装tensorflow：

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==2.0.0-beta1

安装tensorflow_datasets：

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow_datasets

## 导入各种依赖包

```python
import tensorflow as tf

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt
```

## 下载Oxford-IIIT Pets数据集

数据集已包含在TensorFlow数据集中，只需下载即可。分段掩码包含在3.0.0版中，这就是使用此特定版本的原因。

```python
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)
```

以下代码执行翻转图像的简单扩充。另外，图像归一化为[0,1]。
最后，如上所述，分割掩模中的像素标记为{1,2,3}。为了方便起见，让我们从分割掩码中减去1，得到标签：{0,1,2}。

```python
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32)/128.0 - 1
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
```

数据集已包含测试和训练所需的分割，因此让我们继续使用相同的分割。

```python
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
```

让我们看一下图像示例，它是数据集的相应掩模。

```python
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
```

![](https://www.tensorflow.org/beta/tutorials/images/segmentation_files/output_a6u_Rblkteqb_0.png)


## 定义模型

这里使用的模型是一个改进的U-Net。U-Net由编码器（下采样器）和解码器（上采样器）组成。为了学习鲁棒特征并减少可训练参数的数量，可以使用预训练模型作为编码器。因此，该任务的编码器将是预训练的MobileNetV2模型，其中间输出将被使用，并且解码器是已经在[Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)教程示例中实现的上采样块。

输出三个通道的原因是因为每个像素有三种可能的标签。可以将其视为多分类，其中每个像素被分为三类。

```python
OUTPUT_CHANNELS = 3
```

如上所述，编码器将是一个预训练的MobileNetV2模型，它已经准备好并可以在[tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications)中使用。编码器由模型中间层的特定输出组成。
请注意，在训练过程中不会训练编码器。

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
```

解码器/上采样器只是在TensorFlow示例中实现的一系列上采样块。

```python
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels):

  # 这是模型的最后一层
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # 通过该模型进行下采样
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
```

## 训练模型

现在，剩下要做的就是编译和训练模型。这里使用的损失是`loss.sparse_categorical_crossentropy`。使用此丢失函数的原因是因为网络正在尝试为每个像素分配标签，就像多类预测一样。在真正的分割掩码中，每个像素都有{0,1,2}。这里的网络输出三个通道。基本上，每个频道都试图学习预测一个类，而 `loss.sparse_categorical_crossentropy` 是这种情况的推荐损失。使用网络输出，分配给像素的标签是具有最高值的通道。这就是create_mask函数正在做的事情。

```python
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

让我们试试模型，看看它在训练前预测了什么。

```python
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()
```

![](https://www.tensorflow.org/beta/tutorials/images/segmentation_files/output_X_1CC0T4dho3_0.png)


让我们观察模型在训练时如何改进。要完成此任务，下面定义了回调函数。

```python
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])
```

![](https://www.tensorflow.org/beta/tutorials/images/segmentation_files/output_StKDH_B9t4SD_0.png)


我们查看损失变化情况
```python
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
```

![](https://www.tensorflow.org/beta/tutorials/images/segmentation_files/output_P_mu0SAbt40Q_0.png)


## 作出预测

让我们做一些预测。为了节省时间，周期的数量很小，但您可以将其设置得更高以获得更准确的结果。

```python
show_predictions(test_dataset, 1)
```

预测效果：
![](https://www.tensorflow.org/beta/tutorials/images/segmentation_files/output_ikrzoG24qwf5_0.png)


## 下一步

现在您已经了解了图像分割是什么，以及它是如何工作的，您可以尝试使用不同的中间层输出，甚至是不同的预训练模型。您也可以通过尝试在Kaggle上托管的[Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge/overview)图像掩蔽比赛来挑战自己。

您可能还希望查看[Tensorflow Object Detection API]（https://github.com/tensorflow/models/tree/master/research/object_detection），以获取您可以重新训练自己数据的其他模型。
