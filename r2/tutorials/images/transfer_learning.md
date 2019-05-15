
# 使用预训练的CNN模型进行迁移学习

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/images/transfer_learning"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

在本章节中，您将学习如何使用预训练网络中的迁移学习对猫与狗图像进行分类。

预训练模型是一个保存的网路，以前在大型数据集上训练的，通常是在大规模图像分类任务上，您可以按原样使用预训练模型，也可以使用转移学习将此模型自定义为给定的任务。

转移学习背后的直觉是，如果一个模型在一个大而且足够通用的数据集上训练，这个模型将有效地作为视觉世界的通用模型。然后，您可以利用这些学习的特征映射，而无需从头开始训练大型数据集上的大型模型。

在本节中，您将尝试两种方法来自定义预训练模型：
1. **特征提取**：使用先前网络学习的表示从新样本中提取有意义的特征，您只需在与训练模型的基础上添加一个新的分类器（将从头开始训练），以便您可以重新调整先前为我们的数据集学习的特征映射。
您不需要(重新)训练整个模型，基本卷积网络已经包含了一些对图片分类非常有用的特性。然而，预训练模型的最后一个分类部分是特定于原始分类任务的，然后是特定于模型所训练的一组类。

2. **微调**：解冻冻结模型的顶层，并共同训练新添加的分类器和基础模型的最后一层，这允许我们“微调”基础模型中的高阶特征表示，以使它们与特定任务更相关。

你将要遵循一般的机器学习工作流程：
1. 检查并理解数据
2. 构建输入管道，在本例中使用Keras 的 `ImageDataGenerator`
3. 构建模型
    * 加载我们的预训练基础模型（和预训练的权重）
    * 将我们的分类图层堆叠在顶部
4. 训练模型
5. 评估模型


```
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

keras = tf.keras
```

## 1. 数据预处理

### 1.1. 下载数据

使用 [TensorFlow Datasets](http://tensorflow.google.cn/datasets)加载猫狗数据集。`tfds` 包是加载预定义数据的最简单方法，如果您有自己的数据，并且有兴趣使用TensorFlow进行导入，请参阅[加载图像数据](https://tensorflow.google.cn/alpha/tutorials/load_data/images)。


```
import tensorflow_datasets as tfds
```

`tfds.load`方法下载并缓存数据，并返回`tf.data.Dataset`对象，这些对象提供了强大、高效的方法来处理数据并将其传递到模型中。

由于`"cats_vs_dog"` 没有定义标准分割，因此使用subsplit功能将其分为训练80%、验证10%、测试10%的数据。

```
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)
```

生成的`tf.data.Dataset`对象包含（图像，标签）对。图像具有可变形状和3个通道，标签是标量。

```
print(raw_train)
print(raw_validation)
print(raw_test)
```

```
    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <DatasetV1Adapter shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
```

显示训练集中的前两个图像和标签：

```
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
```


![png](https://tensorflow.google.cn/alpha/tutorials/images/transfer_learning_files/output_14_0.png)

![png](https://tensorflow.google.cn/alpha/tutorials/images/transfer_learning_files/output_14_1.png)


### 1.2. 格式化数据

使用`tf.image`模块格式化图像，将图像调整为固定的输入大小，并将输入通道重新调整为`[-1,1]`范围。

<!-- TODO(markdaoust): fix the keras_applications preprocessing functions to work in tf2 -->

```
IMG_SIZE = 160 # 所有图像将被调整为160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label
```

使用map方法将此函数应用于数据集中的每一个项：

```
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
```

打乱和批处理数据：

```
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
```

检查一批数据：

```
for image_batch, label_batch in train_batches.take(1):
  pass

image_batch.shape
```

```
    TensorShape([32, 160, 160, 3])
```

## 2. 从预先训练的网络中创建基础模型

您将从Google开发的**MobileNet V2**模型创建基础模型，这是在ImageNet数据集上预先训练的，一个包含1.4M图像和1000类Web图像的大型数据集。ImageNet有一个相当随意的研究训练数据集，其中包括“jackfruit(菠萝蜜)”和“syringe(注射器)”等类别，但这个知识基础将帮助我们将猫和狗从特定数据集中区分开来。

首先，您需要选择用于特征提取的MobileNet V2层，显然，最后一个分类层（在“顶部”，因为大多数机器学习模型的图表从下到上）并不是非常有用。相反，您将遵循通常的做法，在展平操作之前依赖于最后一层，该层称为“瓶颈层”，与最终/顶层相比，瓶颈层保持了很多通用性。

然后，实例化预装了ImageNet上训练的MobileNet V2模型权重，通过制定include_top=False参数，可以加载不包含顶部分类层的网络，这是特征提取的理想选择。

```
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# 从预先训练的模型MobileNet V2创建基础模型 
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

此特征提取器将每个160x160x3图像转换为5x5x1280的特征块，看看它对示例批量图像的作用：

```
feature_batch = base_model(image_batch)
print(feature_batch.shape)
```

```
    (32, 5, 5, 1280)
```


## 3. 特征提取

您将冻结上一步创建的卷积基，并将其用作特征提取器，在其上添加分类器并训练顶级分类器。

### 3.1. 冻结卷积基

在编译和训练模型之前，冻结卷积基是很重要的，通过冻结（或设置`layer.trainable = False`），可以防止在训练期间更新给定图层中的权重。MobileNet V2有很多层，因此将整个模型的可训练标志设置为`False`将冻结所有层。


```
base_model.trainable = False
base_model.summary() # 看看基础模型架构  
```

```
    Model: "mobilenetv2_1.00_160"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            [(None, 160, 160, 3) 0
    __________________________________________________________________________________________________
    Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_1[0][0]
    __________________________________________________________________________________________________
    Conv1 (Conv2D)                  (None, 80, 80, 32)   864         Conv1_pad[0][0]
    __________________________________________________________________________________________________
    .....（此处省略很多层）
    __________________________________________________________________________________________________
    Conv_1_bn (BatchNormalizationV1 (None, 5, 5, 1280)   5120        Conv_1[0][0]
    __________________________________________________________________________________________________
    out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]
    ==================================================================================================
    ...
```


### 3.2. 添加分类头

要从特征块生成预测，请用5x5在空间位置上进行平均，使用`tf.keras.layers.GlobalAveragePooling2D`层将特征转换为每个图像对应一个1280元素向量。

```
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
```

`(32, 1280)`


应用`tf.keras.layers.Dense`层将这些特征转换为每个图像的单个预测。您不需要激活函数，因为此预测将被视为`logit`或原始预测值。正数预测第1类，负数预测第0类。

```
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
```

``` 
    (32, 1)
```


现在使用`tf.keras.Sequential`堆叠特征提取器和这两个层：

```
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```

### 3.3. 编译模型

你必须在训练之前编译模型，由于有两个类，因此使用二进制交叉熵损失：

```
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
model.summary()
```
```
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    mobilenetv2_1.00_160 (Model) (None, 5, 5, 1280)        2257984
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 1280)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 1281
    =================================================================
    Total params: 2,259,265
    Trainable params: 1,281
    Non-trainable params: 2,257,984
    _________________________________________________________________
```

MobileNet中的2.5M参数被冻结，但Dense层中有1.2K可训练参数，它们分为两个`tf.Variable`对象：权重和偏差。


```
len(model.trainable_variables)
```
`2`



### 3.4. 训练模型

经过10个周期的训练后，你应该看到约96%的准确率。

<!-- TODO(markdaoust): delete steps_per_epoch in TensorFlow r1.14/r2.0 -->


```
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)

initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
```

```
    20/20 [==============================] - 4s 219ms/step - loss: 3.1885 - accuracy: 0.6109
```



```
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
```

```
    initial loss: 3.19
    initial accuracy: 0.61
```



```
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
```

```
    Epoch 1/10
    581/581 [==============================] - 102s 175ms/step - loss: 1.8917 - accuracy: 0.7606 - val_loss: 0.8860 - val_accuracy: 0.8828
    ...
    Epoch 10/10
    581/581 [==============================] - 96s 165ms/step - loss: 0.4921 - accuracy: 0.9381 - val_loss: 0.1847 - val_accuracy: 0.9719
```

### 3.5. 学习曲线

让我们来看一下使用MobileNet V2基础模型作为固定特征提取器时，训练和验证准确性/损失的学习曲线。

```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](https://tensorflow.google.cn/alpha/tutorials/images/transfer_learning_files/output_50_0.png)


*注意：如果您想知道为什么验证指标明显优于训练指标，主要因素是因为像`tf.keras.layers.BatchNormalization`和`tf.keras.layers.Dropout`这样的层会影响训练期间的准确性。在计算验证损失时，它们会被关闭。*

在较小程度上，这也是因为训练指标报告了一个周期的平均值，而验证指标是在周期之后进行评估的，因此验证指标会看到已经训练稍长一些的模型。

## 4. 微调

在我们的特征提取实验中，您只在MobileNet V2基础模型上训练了几层，训练期间未预先更新预训练网络的权重。

进一步提高性能的方法是训练（或“微调”）预训练模型的顶层的权重以及您添加的分类器的训练，训练过程将强制将权重通过特征图调整为专门与我们的数据集关联的特征。

*注意：只有在训练顶级分类器并将预先训练的模型设置为不可训练之后，才应尝试此操作。如果您在预先训练的模型上添加一个随机初始化的分类器并尝试联合训练所有层，则梯度更新的幅度将太大（由于分类器的随机权重），并且您的预训练模型将忘记它学到的东西。*

此外，您应该尝试微调少量顶层而不是整个MobileNet模型，在大多数卷积网络中，层越高，它就越专业化。前几层学习非常简单和通用的功能，这些功能可以推广到几乎所有类型的图像，随着层越来越高，这些功能越来越多地针对训练模型的数据集。微调的目的是使这些专用功能适应新数据集，而不是覆盖通用学习。

### 4.1. 取消冻结模型的顶层


您需要做的就是解冻`base_model`并将底层设置为无法训练，然后重新编译模型（这些更改生效所必须的），并恢复训练。


```
base_model.trainable = True

# 看看基础模型有多少层 
print("Number of layers in the base model: ", len(base_model.layers))

# 从此层开始微调 
fine_tune_at = 100

# 冻结‘fine_tune_at’层之前的所有层
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
```
```
    Number of layers in the base model:  155
```

### 4.2. 编译模型

使用低得多的训练率（学习率）编译模型：

```
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
              
model.summary()
```
```
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    mobilenetv2_1.00_160 (Model) (None, 5, 5, 1280)        2257984
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 1280)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 1281
    =================================================================
    Total params: 2,259,265
    Trainable params: 1,863,873
    Non-trainable params: 395,392
    _________________________________________________________________
```


```
len(model.trainable_variables)
```

``` 
   58
```



### 4.3. 继续训练模型

如果你训练得更早收敛，这将使你的准确率提高几个百分点。

```
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch = initial_epochs,
                         validation_data=validation_batches)
```
```
    ...
    Epoch 20/20
    581/581 [==============================] - 116s 199ms/step - loss: 0.1243 - accuracy: 0.9849 - val_loss: 0.1121 - val_accuracy: 0.9875
```

让我们看一下训练和验证精度/损失的学习曲线，当微调MobileNet V2基础模型的最后几层并在其上训练分类器是，验证损失远远高于训练损失，因此您可能有一些过度拟合。因为新的训练集相对较小且与原始的MobileNet V2数据集类似。

经过微调后，模型精度几乎达到98%。

```
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

![png](https://tensorflow.google.cn/alpha/tutorials/images/transfer_learning_files/output_67_0.png)


## 5. 小结:

* **使用预训练的模型进行特征提取：**
使用小型数据集时，通常会利用在同一域中的较大数据集上训练的模型所学习的特征。这是通过实例化预先训练的模型，并在顶部添加完全连接的分类器来完成的。预训练的模型被“冻结”并且仅在训练期间更新分类器的权重。在这种情况下，卷积基提取了与每幅图像相关的所有特征，您只需训练一个分类器，根据所提取的特征集确定图像类。

* **微调与训练的模型：** 
为了进一步提高性能，可以通过微调将预训练模型的顶层重新调整为新数据集。在这种情况下，您调整了权重，以便模型学习特定于数据集的高级特征，当训练数据集很大并且非常类似于预训练模型训练的原始数据集时，通常建议使用此技术。

