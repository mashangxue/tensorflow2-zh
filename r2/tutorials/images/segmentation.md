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


```
!pip install git+https://github.com/tensorflow/examples.git
```


```
!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf
```


```
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt
```

## 下载Oxford-IIIT Pets数据集

数据集已包含在TensorFlow数据集中，只需下载即可。分段掩码包含在3.0.0版中，这就是使用此特定版本的原因。

```
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)
```

以下代码执行翻转图像的简单扩充。另外，图像归一化为[0,1]。最后，如上所述，分割掩模中的像素标记为{1,2,3}。为了方便起见，让我们从分割掩码中减去1，得到标签：{0,1,2}。

```
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32)/128.0 - 1
  input_mask -= 1
  return input_image, input_mask
```


```
@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
```


```
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
```

数据集已包含测试和训练所需的分割，因此让我们继续使用相同的分割。

```
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
```


```
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)
```


```
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
```

让我们看一下图像示例，它是数据集的相应掩模。

```
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
```


```
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
```

## 定义模型
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 
这里使用的模型是一个改进的U-Net。U-Net由编码器（下采样器）和解码器（上采样器）组成。为了学习鲁棒特征并减少可训练参数的数量，可以使用预训练模型作为编码器。因此，该任务的编码器将是预训练的MobileNetV2模型，其中间输出将被使用，并且解码器是已经在Pix2pix教程示例中实现的上采样块。

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.


```
OUTPUT_CHANNELS = 3
```

As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.


```
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

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
```

The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.


```
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]
```


```
def unet_model(output_channels):

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
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

## Train the model
Now, all that is left to do is to compile and train the model. The loss being used here is losses.sparse_categorical_crossentropy. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and losses.sparse_categorical_crossentropy is the recommended loss for such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.


```
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Let's try out the model to see what it predicts before training.


```
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
```


```
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
```


```
show_predictions()
```

Let's observe how the model improves while it is training. To accomplish this task, a callback function is defined below. 


```
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
```


```
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])
```


```
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

## Make predictions

Let's make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.


```
show_predictions(test_dataset, 3)
```

## Next steps
Now that you have an understanding of what image segmentation is and how it works, you can try this tutorial out with different intermediate layer outputs, or even different pretrained model. You may also challenge yourself by trying out the [Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge/overview) image masking challenge hosted on Kaggle.

You may also want to see the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for another model you can retrain on your own data.
