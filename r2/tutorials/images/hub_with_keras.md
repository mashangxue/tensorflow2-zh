
# TensorFlow Hub with Keras

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/images/hub_with_keras"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/images/hub_with_keras.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/images/hub_with_keras.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

[TensorFlow Hub](http://tensorflow.org/hub)是一种共享预训练模型组件的方法。有关预先训练模型的可搜索列表，请参阅[TensorFlow模块中心TensorFlow Module Hub](https://tfhub.dev/)。

本教程演示：
1. 如何在tf.keras中使用TensorFlow Hub。
1. 如何使用TensorFlow Hub进行图像分类。
1. 如何做简单的迁移学习。

## 1. 安装和导入包

安装命令：`pip install -U tensorflow_hub`

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt

import tensorflow as tf
 
import tensorflow_hub as hub

from tensorflow.keras import layers
```

## 2. ImageNet分类器

### 2.1. 下载分类器

使用`hub.module`加载mobilenet，并使用`tf.keras.layers.Lambda`将其包装为keras层。
来自tfhub.dev的任何兼容tf2的[图像分类器URL](https://tfhub.dev/s?q=tf2&module-type=image-classification)都可以在这里工作。

```python
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])
```

### 2.2. 在单个图像上运行它

下载单个图像以试用该模型。

```python
import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape
```
`(224, 224, 3)`

添加批量维度，并将图像传递给模型。

```python
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape
```

结果是1001元素向量的`logits`，对图像属于每个类的概率进行评级。因此，可以使用`argmax`找到排在最前的类别ID：

```python
predicted_class = np.argmax(result[0], axis=-1)
predicted_class
```
```
653
```

### 2.3. 解码预测


我们有预测的类别ID，获取`ImageNet`标签，并解码预测

```python
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```
![png](https://tensorflow.google.cn/alpha/tutorials/images/hub_with_keras_files/output_20_0.png)

## 3. 简单的迁移学习

使用TF Hub可以很容易地重新训练模型的顶层以识别数据集中的类。

### 3.1. Dataset

对于此示例，您将使用TensorFlow鲜花数据集：

```python
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
```

将此数据加载到我们的模型中的最简单方法是使用 `tf.keras.preprocessing.image.ImageDataGenerator`,

所有TensorFlow Hub的图像模块都期望浮点输入在“[0,1]”范围内。使用`ImageDataGenerator`的`rescale`参数来实现这一目的。图像大小将在稍后处理。

```python
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
```

```
    Found 3670 images belonging to 5 classes.
```
结果对象是一个返回`image_batch，label_batch`对的迭代器。

```python
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break
```

```
    Image batch shape:  (32, 224, 224, 3)
    Labe batch shape:  (32, 5)
```

### 3.2. 在一批图像上运行分类器

现在在图像批处理上运行分类器。


```python
result_batch = classifier.predict(image_batch)
result_batch.shape  # (32, 1001)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names
```

```
      array(['daisy', 'sea urchin', 'ant', 'hamper', 'daisy', 'ringlet',
             'daisy', 'daisy', 'daisy', 'cardoon', 'lycaenid', 'sleeping bag',
             'Bedlington terrier', 'daisy', 'daisy', 'picket fence',
             'coral fungus', 'daisy', 'zucchini', 'daisy', 'daisy', 'bee',
             'daisy', 'daisy', 'bee', 'daisy', 'picket fence', 'bell pepper',
             'daisy', 'pot', 'wolf spider', 'greenhouse'], dtype='<U30')
```

现在检查这些预测如何与图像对齐：

```python
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
```

![png](https://tensorflow.google.cn/alpha/tutorials/images/hub_with_keras_files/output_34_0.png)

有关图像属性，请参阅`LICENSE.txt`文件。

结果没有那么完美，但考虑到这些不是模型训练的类（“daisy雏菊”除外），这是合理的。

### 3.3. 下载无头模型

TensorFlow Hub还可以在没有顶级分类层的情况下分发模型。这些可以用来轻松做迁移学习。

来自tfhub.dev的任何[Tensorflow 2兼容图像特征向量URL](https://tfhub.dev/s?module-type=image-feature-vector&q=tf2)都可以在此处使用。

```python
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
```

创建特征提取器。

```python
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
```

它为每个图像返回一个1280长度的向量：

```python
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
```
`(32, 1280)`

冻结特征提取器层中的变量，以便训练仅修改新的分类器层。

```python
feature_extractor_layer.trainable = False
```

### 3.4. 附上分类头

现在将中心层包装在`tf.keras.Sequential`模型中，并添加新的分类层。

```python
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])

model.summary()
```
```
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer_1 (KerasLayer)   (None, 1280)              2257984   
    _________________________________________________________________
    dense (Dense)                (None, 5)                 6405      
    =================================================================
    Total params: 2,264,389
    Trainable params: 6,405
    Non-trainable params: 2,257,984
    _________________________________________________________________
```

```python
predictions = model(image_batch)
predictions.shape
```
```
    TensorShape([32, 5])
```

### 3.5. 训练模型

使用compile配置训练过程：

```python
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])
```

现在使用`.fit`方法训练模型。

这个例子只是训练两个周期。要显示训练进度，请使用自定义回调单独记录每个批次的损失和准确性，而不是记录周期的平均值。

```python
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats_callback])
```

```
    Epoch 1/2
    115/115 [==============================] - 22s 193ms/step - loss: 0.8613 - acc: 0.8438
    Epoch 2/2
    115/115 [==============================] - 23s 199ms/step - loss: 0.5083 - acc: 0.7812
```

现在，即使只是几次训练迭代，我们已经可以看到模型正在完成任务。

```python
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
```

![png](https://tensorflow.google.cn/alpha/tutorials/images/hub_with_keras_files/output_53_1.png)

```python
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
```
![png](https://tensorflow.google.cn/alpha/tutorials/images/hub_with_keras_files/output_54_1.png?dcb_=0.5728569869098554)

### 3.6. 检查预测

要重做之前的图，首先获取有序的类名列表：

```python
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names
```
```
    array(['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'],
          dtype='<U10')
```

通过模型运行图像批处理，并将索引转换为类名。

```python
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
```

绘制结果

```python
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
```

![png](https://tensorflow.google.cn/alpha/tutorials/images/hub_with_keras_files/output_61_0.png)

## 4. 导出你的模型

现在您已经训练了模型，将其导出为已保存的模型：

```python
import time
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)

export_path
```
```
'/tmp/saved_models/1557794138'
```

现在确认我们可以重新加载它，它仍然给出相同的结果：

```python
reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()
```
`0.0`

这个保存的模型可以在以后加载推理，或转换为[TFLite](https://www.tensorflow.org/lite/convert/) 和 [TFjs](https://github.com/tensorflow/tfjs-converter)。

