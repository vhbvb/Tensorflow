简介：
--
TensorFlow™ 是一个开放源代码软件库，用于进行高性能数值计算。借助其灵活的架构，用户可以轻松地将计算工作部署到多种平台（CPU、GPU、TPU）和设备（桌面设备、服务器集群、移动设备、边缘设备等）。TensorFlow™ 最初是由 Google Brain 团队（隶属于 Google 的 AI 部门）中的研究人员和工程师开发的，可为机器学习和深度学习提供强力支持，并且其灵活的数值计算核心广泛应用于许多其他科学领域。

安装：
--

```
pip install tensorflow
```

TensorFlow中的 "hello world"
--

#### 热身

- Python 程序生成了一些三维数据, 然后用一个平面拟合它. 及训练过程的简化示例

```python
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点. 
x_data = np.float32(np.random.rand(2, 100)) # 你可以理解为这个 训练输入集
y_data = np.dot([0.100, 0.200], x_data) + 0.300 # 你可以理解为这个是 训练集正确的结果

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 用梯度下降来创建优化器，下降速率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 最小化损失函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
sess.run(train)
if step % 20 == 0:
print step, sess.run(W), sess.run(b)
```

- 示例board: [完整示例board](https://google-developers.gonglchuangl.net/machine-learning/crash-course/playground/?utm_source=engedu&utm_medium=ss&utm_campaign=mlcc&hl=zh-cn#activation=relu&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.1&regularizationRate=0.01&noise=80&networkShape=3,2&seed=0.38953&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&tutorial=dp-neural-net-intro-spiral&problem=classification&initZero=false&hideText=true&goalTestLossMinThresholdFirst=0.4&goalTestLossMinThresholdSecond=0.25&problem_hide=true)

#### 机器学习的入门：MNIST

-  **概念：**
MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片

![image](http://www.tensorfly.cn/tfdoc/images/MNIST.png)

它也包含每一张图片对应的标签，告诉我们这个是数字几。比如，上面这四张图片的标签分别是5，0，4，1。
下面将介绍如何用TensorFlow训练一个机器学习模型用于预测图片里面的数字。（目的不是要设计一个世界一流的复杂模型，而是要介绍下如何使用TensorFlow）

对应这个教程的实现代码很短，而且真正有意思的内容只包含在三行代码里面。但是，去理解包含在这些代码里面的设计思想是非常重要的：TensorFlow工作流程和机器学习的基本概念

-  **数据集**（[下载地址](https://www.kaggle.com/c/digit-recognizer/data)）：

下载下来是csv文件，下面解释这些数据源是啥意思：

我们理解的数据源应该是一张图，只是此数据集将手写图转换为矩阵形式

![image](http://www.tensorfly.cn/tfdoc/images/MNIST-Matrix.png)

图片包含28像素X28像素。我们可以用一个数字数组来表示这张图片，
数组展开成一个向量，长度是 28x28 = 784，那么表示一张图片就可以用 1*784的矩阵表示，如果第一列加上图片代表的数字，那么一组训练单元就是 1 * 785的矩阵，那么60000组图片和结果集组成的训练集 就是 60000 * 785 的矩阵格式，即我们看到的csv里面的数据

- **模型的设计**

- 使用一个最简单的单层的神经网络进行学习

- 用SoftMax来做为激活函数

- 用交叉熵来做损失函数

- 用梯度下降来做优化方式

- **名词解释:**

- 神经网络：

很多个神经元组成，每个神经元接收很多个输入：[X1,X2....Xn]，加权相加然后加上偏移量后，看是不是超过了某个阀值，超过了发出1，没超过发出0。

形象的介绍： xxx的喜欢妹子的标准， 比如决定他喜欢一个妹子有很多因素,即特征，比如 颜值，三围，身高，发色，长发短发等, 那么 对妹子这些特征需要分类器进行分类，输出为喜欢（1），或者不喜欢（0），
对于这种把特征空间一分为2的分类器 即为神经元

这个模型有点像人脑中的神经元：从多个感受器接受电信号，x1,x2,x3...xn ，进行处理（加权相加再偏移一点，即判断输入是否在某条直线的一侧），发出电信号（在正确的那侧发出1，否则不发信号，可以认为是发出0），这就是它叫神经元的原因。

但神经元缺点只能简单的一分为二，没法分割复杂特征分布的两类
解决办法是多层神经网络，底层神经元的输出是高层神经元的输入。

```
graph TD
A[一群妹子]--> B[颜值特征]
A --> C[三围特征]
A --> D[发色特征]
B --> Z[特征1]
C --> Z
D --> Z
B --> Y[特征2]
C--> Y
D-->Y
Z --> E{激活函数}
Y --> E
E --> F[喜欢]
E --> G[不喜欢]

```

**隐藏层实现：**

![cdfa594b-c3b8-48e9-8f42-50f38deb117d.png](http://note.youdao.com/yws/res/8511///note.youdao.com/src/WEBRESOURCEbeae9f1486c12d26a89a2b8a8f9d0ee4)

- **激活函数**：每个神经元，在通过一系列计算后，得到了一个数值，怎么来判断应该输出什么呢？激活函数就是解决这个问题，你把值给我，我来判断怎么输出

- **SoftMax**：我们知道max(A,B)，指A和B里哪个大就取哪个值，但我们有时候希望比较小的那个也有一定概率取到，怎么办呢？我们就按照两个值的大小，计算出概率，按照这个概率来取A或者B。比如A=9，B=1,那取A的概率是90%，取B的概率是10%。

- **损失函数**：损失函数是模型对数据拟合程度的反映，拟合得越好损失应该越小，拟合越差损失应该越大，然后我们根据损失函数的结果对模型进行调整。

- **交叉熵**: 预测的概率分布 和 实际的分布的预测, 是描述真相的低效性，训练的过程就是最小化低效性的过程

- **梯度下降**：梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动

神经网络的训练依靠反向传播算法：最开始输入层输入特征向量，网络层层计算获得输出，输出层发现输出和正确的类号不一样，这时它就让最后一层神经元进行参数调整，最后一层神经元不仅自己调整参数，还会勒令连接它的倒数第二层神经元调整，层层往回退着调整。经过调整的网络会在样本上继续测试，如果输出还是老分错，继续来一轮回退调整，直到网络输出满意为止。

TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。

- **代码**：

```python

#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd 

#加载训练集
train = pd.read_csv("data/train.csv")

#处理图片数据集
images = train.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0/255.0)

image_size = images.shape[1]
# image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
x = tf.placeholder("float", shape=[None, image_size])

#结果label集
labels_flat = train.iloc[:,:1].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
y = tf.placeholder("float", shape=[None, labels_count])

#独热编码
def dense_to_one_hot(label_dense,num_classes):
num_lables = label_dense.shape[0]
index_offset = np.arange(num_lables) *num_classes
label_one_hot = np.zeros((num_lables, num_classes))
label_one_hot.flat[index_offset + label_dense.ravel()] = 1
return label_one_hot

labels = dense_to_one_hot(labels_flat,labels_count)
labels = labels.astype(np.uint8)

# 分组数据集 为训练组 和 验证组
VALIDATION_SIZE = 2000
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# 创建模型
weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x,weights)+biases
prediction = tf.nn.softmax(result)

# 损失函数是模型对数据拟合程度的反映，拟合得越好损失应该越小，拟合越差损失应该越大，然后我们根据损失函数的结果对模型进行调整。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels =y, logits= prediction))

# 自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。‘
# 在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 创建的变量初始化器
init = tf.global_variables_initializer()

# 创建session 并且初始化变量
with tf.Session() as sess:

sess.run(init)

#切片读取
batch_size = 100
n_bacth = train_images.shape[0]/batch_size

#重复训练50次 正确率是92%左右
for epoch in range(50):

for bacth in range(n_bacth):
batch_x = train_images[bacth*batch_size:(bacth + 1)*batch_size]
bacth_y = train_labels[bacth*batch_size:(bacth + 1)*batch_size]
sess.run(train_step,feed_dict={x:batch_x, y:bacth_y})

accuracy_n = sess.run(accuracy,feed_dict={x: validation_images, y: validation_labels}) 
print("第" + str(epoch+1) + "轮，准确度为：" + str(accuracy_n))

```


参考文档
--
- [人工神经网络中究竟使用多少隐藏层和神经元](https://www.jianshu.com/p/91138ced2939)
- [机器学习速成课程](https://developers.google.com/machine-learning/crash-course/)
- [tensorfly中文网](http://www.tensorfly.cn/)
- [TensorFlow官网](https://www.tensorflow.org/get_started/)
- [Tensorflow之MNIST解析](https://www.cnblogs.com/lizheng114/p/7439556.html)
- [零基础用TensorFlow玩转Kaggle的“手写识别](https://www.jianshu.com/p/696bde1641d8)
- [五分钟带你入门TensorFlow](https://www.jianshu.com/p/2ea7a0632239)

- [如何简单形象又有趣地讲解神经网络是什么？](https://www.zhihu.com/question/22553761) 
- [损失函数为什么是凸函数](http://mathgotchas.blogspot.com/2011/10/why-is-error-function-minimized-in.html)
- [对卷积神经网络工作原理做一个直观的解释?](https://www.zhihu.com/question/39022858)
- [训练分类自己的图片（CNN超详细入门版）](https://blog.csdn.net/Missayaaa/article/details/79119839)
