# PADDLE学习笔记

课程设计

![](C:\Users\liu12\Desktop\课程目标.png)

课程作业得分情况分布：

![](C:\Users\liu12\Desktop\作业.png)

## 目录

<a href="#1" style="text-decoration:none">一、深度神经网络的算法解析</a>

> <a href="#1.1" style="text-decoration:none">1.1人工智能的分类</a>
>
> <a href="#1.2" style="text-decoration:none">1.2机器学习的类型</a>
>
> <a href="#1.3" style="text-decoration:none">1.3人工智能、机器学习、深度学习的关系</a>
>
> <a href="#1.4" style="text-decoration:none">1.4深度神经网络基础</a> 
>
> > <a href="#1.4.2" style="text-decoration:none">1.4.2神经网络基础知识</a>
> >
> > <a href="#1.4.3" style="text-decoration:none">1.4.3训练过程三部曲</a>
> >
> > > <a href="#1.4.3.1" style="text-decoration:none">正向传播</a>
> > >
> > > <a href="#1.4.3.2" style="text-decoration:none">反向传播</a>
> > >
> > > <a href="#1.4.3.3" style="text-decoration:none">梯度下降</a>

## <h2 id="1">1.深度神经网络的算法解析</h2>

### <h3 id="1.1">1.1人工智能的分类</h3>

弱人工智能：特定任务与人类治理或者效率持平

通用人工智能：具有人类智力水平，解决通用问题

超人工智能：超过人类智力水平，可以在创造力上超过常人

### <h3 id="1.2">1.2机器学习的类型</h3>

有监督学习：通过标签的训练数据集，带有标签（人脸识别）

无监督学习：通过无标签数据集自动发掘模式（文本自聚类）

增强学习：通过反馈或者奖惩机制学习（游戏）

### <h3 id="1.3">1.3人工智能、机器学习、深度学习的关系</h3>

人工智能：研究开发用于模拟延伸和扩展人的智能的理论方法技术及应用系统的一门新的技术科学

机器学习：如果一个程序可以在任务T上，随着经验E的增加，效

深度学习的应用：语音，图像，自然语言处理

深度学习相较传统机器学习算法的优势，深度学习数据规模越大效果越好

![](C:\Users\liu12\Desktop\算法比较.png)

### <h3 id="1.4">1.4深度神经网络基础</h3>

#### <h4 id="1.4.1">1.4.1三个神经网络的基础算法：</h4>

DNN深度神经网络

CNN卷积神经网络

RNN循环神经网络（主要用于语音识别）

#### <h4 id="1.4.2">1.4.2神经网络基础知识</h4>

神经网络我们输入的是一个向量，我们输出的是一个标量

神经元的内部包括两部份运算：

1.线性变换（加权求和）

2.非线性变换（非线性函数）就是激活函数

运算过程：输出=线性变换+非线性变换

判断神经网络的层数：不包括输入层，但是包括输出层

全连接神经网络（FC）：每个神经元都和下一层的所有神经元相连

每一个神经元的输出都是一个标量

#### <h4 id="1.4.3">1.4.3训练过程三部曲</h4>

##### <h5 id="1.4.3.1">正向传播</h5>

输入数据，我们通过神经网络我们最后得到一个最终的结果，这是正向传播的一个过程

包括两个过程一个是线性过程，一个是非线性过程（即激活函数的配置）

常见的**激活函数**

sigmoid(最有名的激活函数）:

特点：x=0函数值是0.5 x越来越大的时候函数的值会接近1，通常做二分类，将一个实数映射到（0，1）

缺点：激活函数计算量大，容易出现梯度消失x过于大或者过于小的时候会出现梯度消失

Tanh函数（双曲线函数）：

特点：上线为1，下限为-1，把数据压缩到-1，1，数据以0为中心

用法：循环神经网络中常用

缺点：计算量过大，也会出现梯度消失的问题

ReLU函数：

特点：大于0的部分输出为数据本身，小于0的部分输出为0，ReLU对于梯度收敛有巨大加速作用，只需要一个阈值就可以得到激活值节省计算量

用法：深层网络中隐藏层常用

缺点：信息容易丢失，大于0的情况下不至于造成梯度消失

**自己配置激活函数的时候，先配置ReLU，再思考其他的方法，目前最常用的仍然是ReLU**

正向传播过程可以理解为网络最终算出一个预测值，通过损失函数来度量预测值与实际值的误差

##### <h5 id="1.4.3.2">反向传播</h5>

对每一个神经元求参数的偏导数

损失函数（cost function/loss function）

均方误差代价函数（线性回归）

平方所形成函数的导数的性质比较好，求和和平均是为了尽量降低由于数据规模所引发的影响。

交叉熵损失函数（二分类）

交叉熵损失函数（多分类问题）

目标是使得通过正向传播所得到的预测值与标签值的差距尽量小

反向传播(BP算法)

由损失函数开始，求每一个神经元中的所有w和b的偏导数

我们只需要配置损失函数即可

##### <h5 id="1.4.3.3">梯度下降</h5>

通过我们刚刚得到的偏导数的值来更新参数值，作用是不断的更新参数，我们的目标是找到参数使得损失函数的值最小

寻找最优参数的方法是梯度下降，梯度下降的核心作用是更新参数，核心思想是不断尝试

梯度下降的过程理解

> 1.初始化所有的w和b（w，b为一个非0且足够小的值，b可以设置为0）
>
> 2.得到损失函数
>
> 3.得到所有参数的偏导数（得到偏导数）
>
> 4.我们得到学习率α（手动设置）
>
> 通常设置为一个0.001等很小的数字
>
> 5.更新w,b

其中的学习率α是超参数，是我们自己所定义的数字，又被称为超参数，学习率不能太大也不能太小，通常取值(0.1，0.001，0.01)，太大：不收敛，太小：学习就会太慢

梯度下降的3种方式

批量梯度下降（BGD）

最原始的方式，在更新每一参数时都是用所有样本来进行更新，

有点：全局最优解，易于并行实现

缺点：当样本数目很多时训练过程就会很慢，每次的计算量都非常大，数据量过大把显存撑爆了

随机梯度下降

每一次从大量样本中随机抽取一个样本，进行迭代

优点：训练速度快

缺点：并不是全局最优，盲目搜索准确度较低，迭代次数增加

小批量梯度下降

结合各自优点，平衡各自缺点，更新参数时使用b个样本（使用2^n次方）一开始设置的要稍微大一点

优点：训练次数尽量小，每次训练的耗时尽量少

训练的整个过程

> 首先输入数据以及标签到神经网络进行正向传播
>
> 接着在反向传播中，我们通过损失函数来判断当前模型的效果如何，接着进行反向传播，即求偏导过程
>
> 最后我们由反向传播当中得到的偏导，进行梯度下降过程，调优参数

## <h2 id="2">2.PADDLEPADDLE线性回归代码实战</h2>

### <h3 id="2.1">2.1线性回归的基本概念</h3>

再来回顾一下线性回归的一些知识：
线性回归是机器学习中最简单也是最重要的模型之一，其模型建立同样遵循上图流程：获取数据、数据预处理、训练模型、应用模型。

回归模型可以理解为：存在一个点集，用一条曲线去拟合它分布的过程。如果拟合曲线是一条直线，则称为线性回归。如果是一条二次曲线，则被称为二次回归。线性回归是回归模型中最简单的一种。

在线性回归中有几个基本的概念需要掌握：

* 假设函数（Hypothesis Function）
* 损失函数（Loss Function）
* 优化算法（Optimization Algorithm）

### <h3 id="2.2">2.2假设函数</h3>

​	假设函数是指，用数学的方法描述自变量和因变量之间的关系，它们之间可以是一个线性函数或非线性函数。 在本次线性回顾模型中，我们的假设函数为 Y^=aX1+b\hat{Y}= aX_1+b*Y*^=*a**X*1+*b* ，其中，Y^\hat{Y}*Y*^表示模型的预测结果（预测房价），用来和真实的Y区分。模型要学习的参数即：a,b。

### <h3 id="2.3">2.3损失函数</h3>

​	损失函数是指，用数学的方法衡量假设函数预测结果与真实值之间的误差。这个差距越小预测越准确，而算法的任务就是使这个差距越来越小。

​	建立模型后，我们需要给模型一个优化目标，使得学到的参数能够让预测值Y^\hat{Y}*Y*^尽可能地接近真实值Y。输入任意一个数据样本的目标值yiy_i*y**i*和模型给出的预测值Yi^\hat{Y_i}*Y**i*^，损失函数输出一个非负的实值。这个实值通常用来反映模型误差的大小。

​	对于线性模型来讲，最常用的损失函数就是均方误差（Mean Squared Error， MSE）。
$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(\hat{Y_i}-Y_i)^2
$$
​	即对于一个大小为n的测试集，MSE是n个数据预测结果误差平方的均值。

### <h3 id="2.4">2.4优化算法</h3>

​	在模型训练中优化算法也是至关重要的，它决定了一个模型的精度和运算速度。本章的线性回归实例中主要使用了梯度下降法进行优化。

​	**梯度下降**是深度学习中非常重要的概念，值得庆幸的是它也十分容易理解。损失函数J*(*w*,*b*)可以理解为变量w和b的函数。观察下图，垂直轴表示损失函数的值，两个水平轴分别表示变量w和b。实际上，可能是更高维的向量，但是为了方便说明，在这里假设w和b*都是一个实数。算法的最终目标是找到损失函数的最小值。而这个寻找过程就是不断地微调变量w和b的值，一步一步地试出这个最小值。而试的方法就是沿着梯度方向逐步移动。本例中让图中的圆点表示损失函数的某个值，那么梯度下降就是让圆点沿着曲面下降，直到取到最小值或逼近最小值。

​	因为是凸函数，所以无论初始化在曲面上的哪一点，最终都会收敛到同一点或者相近的点。

### <h3 id="2.5">2.5代码实战</h3>

#### <h4 id="2.5.1">2.5.1引入库</h4>

```python
#引入库

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt#绘图工具
import pandas as pd
import paddle
import paddle.fluid as fluid #引入fluid版本库

import math
import sys

#%matplotlib inline
```

#### <h4 id="2.5.2">2.5.2数据进行预处理&&归一化处理

​	一般拿到一组数据后，第一个要处理的是数据类型不同的问题。如果各维属性中有离散值和连续值，就必须对离散值进行处理。

​	离散值虽然也常使用类似0、1、2这样的数字表示，但是其含义与连续值是不同的，因为这里的差值没有实际意义。例如，我们用0、1、2来分别表示红色、绿色和蓝色的话，我们并不能因此说“蓝色和红色”比“绿色和红色”的距离更远。通常对有d个可能取值的离散属性，我们会将它们转为d个取值为0或1的二值属性或者将每个可能取值映射为一个多维向量。

​	不过就这里而言，数据中没有离散值，就不用考虑这个问题了。

**归一化**

​	观察一下数据的分布特征，一般而言，如果样本有多个属性，那么各维属性的取值范围差异会很大，这就要用到一个常见的操作-归一化（normalization）了。归一化的目标是把各维属性的取值范围放缩到差不多的区间，例如[-0.5, 0.5]。这里我们使用一种很常见的操作方法：减掉均值，然后除以原取值范围。

​	基本上所有的数据在拿到后都必须进行归一化，至少有以下3条原因：

​	1.过大或过小的数值范围会导致计算时的浮点上溢或下溢。

​	2.不同的数值范围会导致不同属性对模型的重要性不同（至少在训练的初始阶段如此），而这个隐含的假设常常是不合理的。这会对优化的过程造成困难，使训练时间大大加长。

​	3.很多的机器学习技巧/模型（例如L1，L2正则项，向量空间模型-Vector Space Model）都基于这样的假设：所有的属性取值都差不多是以0为均值且取值范围相近的。

```python
# coding = utf-8 #
global x_raw,train_data,test_data
data = np.loadtxt('./datasets/data.txt',delimiter = ',')
x_raw = data.T[0].copy() 

#axis=0,表示按列计算
#data.shape[0]表示data中一共有多少列
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
print("the raw area :",data[:,0].max(axis = 0))

#归一化，data[:,i]表示第i列的元素

### START CODE HERE ### (≈ 3 lines of code)
for i in range(0,data.shape[0]):#此处进行归一化
    data[:,0][i] = (data[:,0][i] - avgs[0])/(maximums[0] - minimums[0])
### END CODE HERE ###

print('normalization:',data[:,0].max(axis = 0))
```

#### <h4 id="2.5.3">2.5.3数据集分割</h4>

将原始数据处理为可用数据后，为了评估模型的好坏，我们将数据分成两份：训练集和测试集。
- 训练集数据用于调整模型的参数，即进行模型的训练，模型在这份数据集上的误差被称为训练误差；
- 测试集数据被用来测试，模型在这份数据集上的误差被称为测试误差。

我们训练模型的目的是为了通过从训练数据中找到规律来预测未知的新数据，所以测试误差是更能反映模型表现的指标。分割数据的比例要考虑到两个因素：更多的训练数据会降低参数估计的方差，从而得到更可信的模型；而更多的测试数据会降低测试误差的方差，从而得到更可信的测试误差。我们这个例子中设置的分割比例为8:2。

```python
#数据集分割
ratio = 0.8
offset = int(data.shape[0]*ratio)

### START CODE HERE ### (≈ 2 lines of code)
train_data = data[:offset]
test_data = data[offset + 1:]

### END CODE HERE ###

print(len(data))
print(len(train_data))
```

#### <h4 id="2.5.4">2.5.4定义reader</h4>

构造read_data()函数，来读取训练数据集train_set或者测试数据集test_set。它的具体实现是在read_data()函数内部构造一个reader()，使用yield关键字来让reader()成为一个Generator（生成器），注意，yield关键字的作用和使用方法类似return关键字，不同之处在于yield关键字可以构造生成器（Generator）。虽然我们可以直接创建一个包含所有数据的列表，但是由于内存限制，我们不可能创建一个无限大的或者巨大的列表，并且很多时候在创建了一个百万数量级别的列表之后，我们却只需要用到开头的几个或几十个数据，这样造成了极大的浪费，而生成器的工作方式是在每次循环时计算下一个值，不断推算出后续的元素，不会创建完整的数据集列表，从而节约了内存使用。

```python
#套路部分
def read_data(data_set):
    """
    一个reader
    Args：
        data_set -- 要获取的数据集
    Return：
        reader -- 用于获取训练集及其标签的生成器generator
    """
    def reader():
        """
        一个reader
        Args：
        Return：
            data[:-1],data[-1:] --使用yield返回生成器
                data[:-1]表示前n-1个元素，也就是训练数据，
                data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            yield data[:-1],data[-1:]
    return reader
```

#### <h4 id="2.5.5">2.5.5数据提取器</h4>

接下来我们定义了用于训练的数据提供器。提供器每次读入一个大小为BATCH_SIZE的数据批次。如果用户希望加一些随机性，它可以同时定义一个批次大小和一个缓存大小。这样的话，每次数据提供器会从缓存中随机读取批次大小那么多的数据。我们都可以通过batch_size进行设置，这个大小一般是2的N次方。

关于参数的解释如下：

* paddle.reader.shuffle(read_data(train_data), buf_size=500)表示从read_data(train_data)中读取了buf_size=500大小的数据并打乱顺序
* paddle.batch(reader(), batch_size=BATCH_SIZE)表示从打乱的数据中再取出BATCH_SIZE=20大小的数据进行一次迭代训练

如果buf_size设置的数值大于数据集本身，就直接把整个数据集打乱顺序；如果buf_size设置的数值小于数据集本身，就按照buf_size的大小打乱顺序。

```python
BATCH_SIZE = 20
print(train_data)
# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(train_data), 
        buf_size=500),
    batch_size=BATCH_SIZE)

#设置测试 reader
test_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(test_data), 
        buf_size=500),
    batch_size=BATCH_SIZE)
```

#### <h4 id="2.5.6">2.5.6训练过程</h4>

训练过程如下

- 配置网络结构和设置参数
    - 配置网络结构
    - 定义损失函数cost
    - 定义执行器(参数随机初始化) 
    - 定义优化器optimizer
- 模型训练
- 预测
- 绘制拟合图像

##### <h5 id="2.5.6.1">2.5.6.1定义运算场所</h5>

​	首先进行最基本的运算场所定义，在 fluid 中使用 place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() 来进行初始化：

* place 表示fluid program的执行设备，常见的有 fluid.CUDAPlace(0) 和 fluid.CPUPlace()
* use_cuda = False 表示不使用 GPU 进行加速训练

```python
#使用CPU或者GPU训练
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() 
```

##### <h5 id="2.5.6.2">2.5.6.2配置网络结构和设置参数</h5>

**配置网络结构**


​	线性回归的模型其实就是一个采用线性激活函数（linear activation）的全连接层（fully-connected layer，fc_layer），因此在Peddlepeddle中利用全连接层模型构造线性回归，这样一个全连接层就可以看做是一个简单的神经网络，只包含输入层和输出层即可。本次的模型由于只有一个影响参数，因此输入只含一个$X_0$。

![](C:\Users\liu12\Desktop\FC.png)

接下来就让我们利用PaddlePaddle提供的接口，搭建我们自己的网络吧！

**输入层**  
我们可以用 x = fluid.layers.data(name='x', shape=[1], dtype='float32')来表示数据的一个输入层，其中name属性的名称为"x"，数据的shape为一维向量，这是因为本次所用的房价数据集的每条数据只有1个属性，所以shape=1。

**输出层**  
用y_predict = fluid.layers.fc(input=x, size=1, act=None)来表示输出层：其中paddle.layer.fc表示全连接层，input=x表示该层出入数据为x，size=1表示该层有一个神经元，在Fluid版本中使用的激活函数不再是调用一个函数了，而是传入一个字符串就可以，比如：act='relu'就表示使用relu激活函数。act=None表示激活函数为线性激活函数。

**标签层**

用y = fluid.layers.data(name='y', shape=[1], dtype='float32')来表示标签数据，名称为y，有时我们名称不用y而用label。数据类型为一维向量。

```python
# 输入层，fluid.layers.data表示数据层,name=’x’：名称为x,输出类型为tensor
# shape=[1]:数据为1维向量
# dtype='float32'：数据类型为float32
### START CODE HERE ### (≈ 1 lines of code)
x = fluid.layers.data(name='x',shape=[1],dtype='float32')

### END CODE HERE ###


# 标签数据，fluid.layers.data表示数据层,name=’y’：名称为y,输出类型为tensor
# shape=[1]:数据为1维向量
### START CODE HERE ### (≈ 1 lines of code)

y = fluid.layers.data(name='y',shape=[1],dtype='float32') 
### END CODE HERE ###

# 输出层，fluid.layers.fc表示全连接层，input=x: 该层输入数据为x
# size=1：神经元个数，act=None：激活函数为线性函数
y_predict = fluid.layers.fc(input=x, size=1, act=None)
```

##### <h5 id="2.5.6.3">2.5.6.3定义损失函数</h5>

```python
# 定义损失函数为均方差损失函数,并且求平均损失，返回值名称为avg_loss
### START CODE HERE ### (≈ 2 lines of code)
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_loss=fluid.layers.mean(cost)
### END CODE HERE ###
```

##### <h5 id="2.5.6.4">2.5.6.4定义执行器</h5>

​	首先定义执行器，fulid使用了一个C++类Executor用于运行一个程序，Executor类似一个解析器，Fluid将会使用这样一个解析器来训练和测试模型。

```python
exe = fluid.Executor(place)
```

##### <h5 id="2.5.6.5">2.5.6.5配置训练程序</h5>

①全局主程序main program。该主程序用于训练模型。

②全局启动程序startup_program。

③测试程序test_program。用于模型测试

```python
main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)
```

##### <h5 id="2.5.6.6">2.5.6.6优化方法</h5>

​	损失函数定义确定后，需要定义参数优化方法。为了改善模型的训练速度以及效果，学术界先后提出了很多优化算法，包括： Momentum、RMSProp、Adam 等，已经被封装在fluid内部，读者可直接调用。本次可以用 fluid.optimizer.SGD(learning_rate= ) 使用随机梯度下降的方法优化，其中learning_rate表示学习率，大家可以自己尝试修改。

```python
# 创建optimizer，更多优化算子可以参考 fluid.optimizer()
learning_rate = 0.01
sgd_optimizer = fluid.optimizer.SGD(learning_rate)
sgd_optimizer.minimize(avg_loss)
print("optimizer is ready")
```

##### <h5 id="2.5.6.7">2.5.6.7训练模型&&创建训练过程</h5>

**训练模型:**

​	上述内容进行了模型初始化、网络结构的配置并创建了训练函数、硬件位置、优化方法，接下来利用上述配置进行模型训练。

**创建训练过程:**

​	训练需要有一个训练程序和一些必要参数，并构建了一个获取训练过程中测试误差的函数。必要参数有executor,program,reader,feeder,fetch_list，executor表示之前创建的执行器，program表示执行器所执行的program，是之前创建的program，如果该项参数没有给定的话则默认使用defalut_main_program，reader表示读取到的数据，feeder表示前向输入的变量，fetch_list表示用户想得到的变量或者命名的结果。

```python
# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失
#定义模型保存路径：
#params_dirname用于定义模型保存路径。
params_dirname = "easy_fit_a_line.inference.model"
```

##### <h5 id="2.5.6.8">2.5.6.8设置主循环</h5>

```python

#用于画图展示训练cost
from paddle.utils.plot import Ploter
train_prompt = "Train cost"
test_prompt = "Test cost"
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

# 训练主循环
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)

exe_test = fluid.Executor(place)

#num_epochs=100表示迭代训练100次后停止训练。
num_epochs = 200

for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0:  # 每10个批次记录并输出一下训练损失
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()
            #print("%s, Step %d, Cost %f" %(train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            #print("%s, Step %d, Cost %f" %(test_prompt, step, test_metics[0]))
            
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)
            
            
            
```

#### <h4 id="2.5.7">2.5.7预测</h4>

**预测**

预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。

* tensor_x:生成batch_size个[0,1]区间的随机数，以 tensor 的格式储存
* results：预测对应 tensor_x 面积的房价结果
* raw_x:由于数据处理时我们做了归一化操作，为了更直观的判断预测是否准确，将数据进行反归一化，得到随机数对应的原始数据。

```python
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets
     ] = fluid.io.load_inference_model(params_dirname, infer_exe) # 载入预训练模型


    batch_size = 2
    tensor_x = np.random.uniform(0, 1, [batch_size, 1]).astype("float32")
    
    print("tensor_x is :" ,tensor_x )
    results = infer_exe.run(
        inference_program,
        feed={feed_target_names[0]: tensor_x},
        fetch_list=fetch_targets) # 进行预测
    raw_x = tensor_x * (maximums[0]-minimums[0])+avgs[0]
    print("the area is:",raw_x)
    print("infer results: ", results[0])

#根据线性模型的原理，补全输出公式，计算a和b的值

#提示：已知两点求直线方程

a = (results[0][0][0] - results[0][1][0]) / (raw_x[0][0]-raw_x[1][0])
b = (results[0][0][0] - a * raw_x[0][0])

print(a,b)
```

#### <h4 id="2.5.8">2.5.8绘制拟合图像</h4>

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_data(data):
    x = data[:,0]
    y = data[:,1]
    y_predict = x*a + b
    plt.scatter(x,y,marker='.',c='r',label='True')
    plt.title('House Price Distributions')
    plt.xlabel('House Area ')
    plt.ylabel('House Price ')
    plt.xlim(0,250)
    plt.ylim(0,2500)
    predict = plt.plot(x,y_predict,label='Predict')
    plt.legend(loc='upper left')
    plt.savefig('result1.png')
    plt.show()

data = np.loadtxt('./datasets/data.txt',delimiter = ',')
plot_data(data)


```



## <h2 id="3"></h2>

softmax只可用于输出层，以及只能用于多分类问题

网络结构设计从高到低

```
h1_=fluid.layers.fc(input=x_,size=32,act='relu')
h2_=fluid.layers.fc(input=h1),size=16,act='relu')
predict_=fluid.layers.fc(input=h2_,size=10,act='softmax')
#多分类问题必须使用softmax，其中的size是表示我们分类的类别总共有多少
```

## <h2 id="3">3.PADLLEPADDLE图像识别代码实战</h2>

### <h3 id="3.1">3.1图片存储知识</h3>

​	在计算机中，图片被存储为三个独立的矩阵，分别对应图3-6中的红、绿、蓝三个颜色通道，如果图片是64*64像素的，就会有三个64*64大小的矩阵，要把这些像素值放进一个特征向量中，需要定义一个特征向量X，将三个颜色通道中的所有像素值都列出来。如果图片是64*64大小的，那么特征向量X的长度就是64*64*3，也就是12288。这样一个长度为12288的向量就是Logistic回归模型的一个训练数

### <h3 id="3.1">3.2引用库</h3>

```python
import sys
import numpy as np

import lr_utils
import matplotlib.pyplot as plt

import paddle
import paddle.fluid as fluid

from paddle.utils.plot import Ploter
%matplotlib inline
```

### <h3 id="3.3">3.3数据预处理

这里简单介绍数据集及其结构。数据集以hdf5文件的形式存储，包含了如下内容：

- 训练数据集：包含了m_train个图片的数据集，数据的标签（Label）分为cat（y=1）和non-cat（y=0）两类。
- 测试数据集：包含了m_test个图片的数据集，数据的标签（Label）同上。

单个图片数据的存储形式为（num_x, num_x, 3），其中num_x表示图片的长或宽（数据集图片的长和宽相同），数字3表示图片为三通道（RGB）。
在代码中使用一行代码来读取数据，读者暂不需要了解数据的读取过程，只需调用load_dataset()方法，并存储五个返回值，以便后续的使用。
    
需要注意的是，为了方便，添加“_orig”后缀表示该数据为原始数据，之后需要对数据做进一步处理。未添加“_orig”的数据则表示之后不需要再对该数据作进一步处理。

```python
# 调用load_dataset()函数，读取数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
#其中lr_utils.load_dataset()返回的是一个多维数组
#这里用到的python语法为num = [[1,2],[3,4],[5,6]]
#a,b,c=num
#a = [1,2]
#b = [3,4]
#c = [5,6]
```

​	获取数据后的下一步工作是获得数据的相关信息，如训练样本个数 m_train、测试样本个数 m_test 和图片的长度或宽度 num_x，使用 numpy.array.shape 来获取数据的相关信息。

查看样本信息:

    - m_train (训练样本数)
    
    - m_test (测试样本数)
    
    - num_px （图片长或宽）

**技巧：**

`train_set_x_orig` 是一个(m_train, num_px, num_px, 3)形状的 numpy 数组。举个例子，你可以使用
    
```python
train_set_x_orig.shape[0]
```

来获得 `m_train`。

```python
m_train =  train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
```

接下来需要对数据作进一步处理，为了便于训练，你可以忽略图片的结构信息，将包含图像长、宽和通道数信息的三维数组压缩成一维数组，图片数据的形状将由(64, 64, 3)转化为(64 * 64 * 3, 1)。


**技巧：**

我们可以使用一个小技巧来将(a,b,c,d)形状的矩阵转化为 (b$*​$c$*​$d, a)形状的矩阵: 

    X_flatten = X.reshape(X.shape[0], -1)

```python
# 定义维度
DATA_DIM = num_px * num_px * 3

# 转换数据形状

### START CODE HERE ### (≈ 2 lines of code) 
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)

```

#### 归一化处理

在开始训练之前，还需要对数据进行归一化处理。图片采用红、绿、蓝三通道的方式来表示颜色，每个通道的单个像素点都存储着一个 0-255 的像素值，所以图片的归一化处理十分简单，只需要将数据集中的每个像素值除以 255 即可，但需要注意的是计算结果应为 float 类型。 现在让我们来归一化数据吧！

```python
#numpy数组直接相除就可以
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255
```

为了方便后续的测试工作，添加了合并数据集和标签集的操作，使用 numpy.hstack 实现 numpy 数组的横向合并。

```python
train_set = np.hstack((train_set_x, train_set_y.T))
test_set = np.hstack((test_set_x, test_set_y.T))
```

**经过上面的实验，大家应该记住:**

对数据进行预处理的一般步骤是:

* 了解数据的维度和形状等信息，例如(m_train, m_test, num_px, ...)
* 降低数据纬度，例如将数据维度(num_px, num_px, 3)转化为(num_px * num_px * 3, 1)
* 数据归一化

### <h3 id="3.4">3 .4 构造reader</h3>

​	构造read_data()函数，来读取训练数据集train_set或者测试数据集test_set。它的具体实现是在read_data()函数内部构造一个reader()，使用yield关键字来让reader()成为一个Generator（生成器），注意，yield关键字的作用和使用方法类似return关键字，不同之处在于yield关键字可以构造生成器（Generator）。虽然我们可以直接创建一个包含所有数据的列表，但是由于内存限制，我们不可能创建一个无限大的或者巨大的列表，并且很多时候在创建了一个百万数量级别的列表之后，我们却只需要用到开头的几个或几十个数据，这样造成了极大的浪费，而生成器的工作方式是在每次循环时计算下一个值，不断推算出后续的元素，不会创建完整的数据集列表，从而节约了内存使用。

```python
# 读取训练数据或测试数据
def read_data(data_set):
    """
        一个reader
        Args:
            data_set -- 要获取的数据集
        Return:
            reader -- 用于获取训练数据集及其标签的生成器generator
    """
    def reader():
        """
        一个reader
        Args:
        Return:
            data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
            data[:-1]表示前n-1个元素组成的list，也就是训练数据，data[1]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            ### START CODE HERE ### (≈ 1 lines of code)
            yield data[:-1],data[-1:]  
            ### END CODE HERE ###
    return reader
```

### <h3 id="3.5">3.5训练工程</h3>

#### 3.5.1 获取训练数据

​	接下来我们定义了用于训练的数据提供器。提供器每次读入一个大小为BATCH_SIZE的数据批次。如果用户希望加一些随机性，可以同时定义一个批次大小和一个缓冲区大小（buf_size）。这样的话，缓冲区中的数据就是乱序的。

​	关于参数的解释如下：

```python
paddle.reader.shuffle(read_data(data_set), buf_size=BUF_SIZE) 
#表示从read_data(data_set)这个reader中读取了BUF_SIZE大小的数据并打乱顺序

paddle.batch(reader(), batch_size=BATCH_SIZE) 
#表示从reader()中取出BATCH_SIZE大小的数据进行一次迭代训练
BATCH_SIZE=200
# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(train_set), buf_size=500),
    batch_size=BATCH_SIZE)
#设置测试 reader
test_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(test_set), buf_size=500),
    batch_size=BATCH_SIZE)
```

SOFTMAX与交叉熵损失函数

softmax用来将我们最后得到的结果映射到一个向量中，其中向量的每一个维度的范围为[0,1]，用来表示数据的归属，最后我们将得到的向量与标签值放入交叉熵损失函数判断损失，逐步调整降低损失。



去除掉数据量纲的影响

防止溢出

加速梯度下降的过程

对于离散值的处理，不用进行归一化

归一化的方法

转化为标准正态分布

最小最大规范化

反向传播概述

反向传播是深度学习的灵魂

反向传播算法是一个通用

梯度消失与梯度爆炸

梯度消失

经过连乘梯度极其小参数几乎不更新

梯度爆炸

b参数过大，无法下山

### 避免梯度消失与梯度爆炸

预训练加微调（迁移学习）

正则化（提高泛化能力）

使用不同的激活函数（Relu,少用sigmoid）

使用batchnorm（批归一化）

使用残差结构（ResNet）

### 多分类分问题

Softmax-----OneHot----交叉熵损失函数

将一个含任意实数的K维向量Z压缩到另一个K维实向量Z中，使得每一个元素的范围都在（0-1）之间，并且所有元素的和为1

一个元素的softmax值表示是该元素的指数与所有元素指数和的比值

独热编码

数组的长度就是类别的数量

每个数组只有某一位是1，其他位都是0

每个数组都为一标识一个类别

表示为第几个类别

#### 多分类系统

输入是图片，输出为独热编码

我们将独热编码经过softmax处理的值视为概率，并按照相对关系，映射到01关系

当有多个输入的时候通过概率求得哪个输入能够胜出



用法

用于多分类神经网络输出层

用于多分类问题的最后一层

缺点

基本只用于多分类问题

但凡是多分类问题我们就会使用交叉熵与softmax来配合完成任务

交叉熵损失函数
$$
H(x)=-\sum_{i=1}^{n}p(x_i)lnP(x_i)
$$
其中H(x)中的x是输入的向量，xi是x的各个分量

softmax中为y<sup>^</sup>

numpy.argsort()返回数组值从小到大的索引值

## reader机制

Reader：

def reader():

​	while True:

​		yield numpy.random.uniform(-1,1,size=width*height0)

## 4.卷积神经网络

### 4.1常用任务

> 图像分类
>
> 目标检测
>
> 语义分割
>
> 实例分割

### 4.2不使用全连接神经网络进行图像分类

内存，计算量巨大，训练十分困难

### 4.3卷积

卷积是两个信号之间的运算

fliter为过滤器

一维卷积有三种方式

- 1.valid卷积核在信号内，顺序相乘求解
- 2.same卷积核中心在信号内，不够的位置（padding）补0，长度不变
- 3.full卷积卷积核边沿在信号内，同样是不够的位置补0

二维卷积

​	此时变成矩阵对应位置相乘求乘积，

多通道卷积

​	对每一个通道进行卷积，最后我们将所有通道卷积后得到的结果进行相加作为最后卷积的结果

卷积核是为了抽取特征

卷积神经网络是为了找到更好的卷积核

越深的卷积所得到的图像就越抽象

放置多少卷积核是超参数

stride反映了filter滑动一次的距离

Padding使卷积后的大小与卷积前的大小相等

输入大小：W1\*H1\*D1

需要指定的超参数:fliter个数(K),fliter大小(F),步长(S),边界补充（P）

输出

W2=（W1-F+2P)/S+1

H2=(H1-F+2P)/S+1

D2=K

 ### 4.4池化层

下降层，把大图变成小层

通道数不发生变化，而是图片的面积发生变化

平均池化与最大池化（最常用）

使得原始图片的尺寸变小了

池化操作时不重叠的，使得面积变为原来的一半 

池化操作没有任何参数需要我们去学习

relu层就是丢掉一些信息

### 4.4过拟合与欠拟合问题

欠拟合模型不能在训练集上取得较小的误差

过拟合：训练误差很小，但是测试误差很大

如果训练误差很小，预测误差也很小就是泛化能力很好

### 4.5容量

容量是指你和各种函数的能力

模型的规模越大，容量也就越大

### 4.6过拟合与欠拟合的解决方案

过拟合：

droupout，正则化，增加数据

欠拟合：
修改参数

## 5.经典卷积神经网络

### 5.1卷积神经网络的一般结构

ReLU非线性单元

使用ReLU非线性函数可以加速训练

局部响应归一化层

现在经常使用的是BN

重叠的池化层

目前的主流方法为size = stride

数据增广

- 将数据经过平移水平翻转
- 随机裁剪
- 色彩抖动
- 加入噪点

来扩充数据

Droupout

- 每个隐藏层的神经元以0.5的概率输出为0.输出为0的神经元相当于从网络中去除。不参与前向计算和反向传播，所以对于每次输入，神经网络都会使用不同的结构
- 测试时需要将Droupout关闭
- 使用BN的时候需要去掉Droupout

 ### 5.2感受野

某一层输出结果中一个元素所对应的输入层的区域大小，被称为感受野

### 5.3小卷积核的优势

在感受野不变的情况下：

一个5*5的卷积核就可以用两个串联的3\*3卷积核来代替

一个7*7的卷积核就可以用三个串联的3\*3卷积核来代替

三个3*3卷积的优势

1.包含三个ReLu层而不是一个，使决策函数更有判别性

2.减少了参数。比如输入输出都是C个通道，使用3*3的三个卷积层需要3（3\*3\*C\*C)=27\*C\*C，使用7\*7的1个卷积层需要7\*7\*C\*C=49C\*C

| 现象       | 效果                           |
| ---------- | ------------------------------ |
| 参数变少   | 计算时间短，存储空间小         |
| 非线性增强 | 拟合能力强                     |
| 层级变高   | 抽象能力强、提取更加复杂的特征 |

### 5.4提升网络性能

最直接的方法就是增加网络深度和宽度，深度指网络层次数量、宽度指神经元数量

问题

- 参数太多，如果训练数据集有限，很容易产生过拟合
- 网络越大、参数越多、计算复杂度越大、难以应用
- 网络越深，容易出现梯度消失问题（梯度越往后越容易消失，难以优化模型

 ### 5.5Inception网络结构

通过设计一个稀疏网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。

1*1卷积的主要目的是为了减少维度，还用于提高非线性（例如ReLU)

 ### 正则化

正则化技术是用来解决模型过拟合问题的

过拟合的常用解决方法

1.让模型看更多的数据

- 数据增强

2.丢弃一部分特征、增强模型的容错能力

- PCA降维
- droupout

3.不丢其特征，保留所有的特征，减少模型参数的大小

- 确保所有的特征对于预测都有所贡献，而不是偏爱某几种特征
- L1
- L2

正则化限制参数过多或者过大，避免模型过于复杂

惩罚总体函数

λ要取合适的范围

## 6.

### 6.1BN（批归一法）

归一化是发生在数据预处理的时候。

常用零-均值归一化方法

BN发生在层与层之间

作用：

- 缓解梯度消失问题
- 加快收敛速度

意义：

- 使得训练深层网络模型更加容易和稳定

BN算法

本质上为两步

- 减去均值，除去方差（标准差）（标准正太分布）
- 数据拉开，平移（还原最初的输入）

可以避免数据处理后两边的数据特征消失的情况，将数据聚拢在中间的位置，使得不丢失数据特征

对于预测阶段时所使用的均值和方差，其实也是来源与训练集。我们在模型训练时我们就记录下每一个batch下的均值和方差，待训练完毕后，我们求整个训练样本的均值和方差期望值作为我们进行预测时进行BN的均值和方差

### 6.2ResNet

#### 6.2.1梯度弥散

梯度消失导致无法对前面网络层的权重进行有效的调整

梯度消失的原因时网络深处的信息难以传到浅层，如果能够将深层的信息回传到浅层，那么这个问题就能解决了

Shortcut的链接方式就可以解决这个问题

每一层的结果我们都加上最开始的输入

### 6.2.2bottleneck design

把输入直接链接到输出上，再进行对应位置的求和，把反向传播过程中导数的连乘变成了连加。

### 6.3迁移学习

让机器自主得从数据中获取知识，从而应用于新的问题中

主要解决问题：

- 缺少大量数据
- 大数据与少标注之间的矛盾
- 大数据与弱计算之间的矛盾
- 普适化模型与个性化需求之间的矛盾

目标与源数据均有标签的情况下，我们采用Fine-tune的方法来进行迁移学习

Fine-tune有两种方案，第一种停止更新卷积层，只更新FC层，第二种是加载原有模型中的w和b

决定因素

- 新旧数据集的大小
- 新旧数据集的相似度

新的数据集很小，并且和原有数据集很相似

采用第一种

新的数据集很大，并且和原有数据集很相似

采用第二种

新的数据集很小，并且和原有数据集不相似

采用第一种

新的数据集很大，并且和原有数据集不相似

采用finetuning

## 7.工程实践

1.分析目前的问题

可用的数据是什么样(高维数据?图片?音频?语料?)

解决方案属于哪类典型问题(分类?回归?)

获取数据 分析数据

尽可能多的获取数据

观察数据的分布,是否存在分布不均匀

观察数据的质量,是否有明显的标签错误,是否有大量人类都无法识别的图片

选择模型.模型调优

确定问题目标:图片分类?目标检测?实例分割?

确定主要约束条件:求准还是求快?在Server还是在嵌入式设备?算力如何?内存是否足够

目前已经是最好效果了吗?还是可以再一进步提高精度?过高的训练精度是否是过拟合了?过高的预测精度是不是会导致泛化能力不足?

上线效果

从系统吐出的错误中查找问题

人脸识别问题,总不能识别的场景,灯光太强,太暗 人脸不正

图片分类问题 把分类错误的图片收集到,仔细观察图片,看是不是标签了?过于模糊?

目标检测问题.某些物体总是识别不出来.是不是目标太小了,太大了,太长了?

 数据预处理

去掉错误的值

采集数据通常可能采集回来错误的值

人工标注也容易标注错误

例如:

工人的工资为负数

标注员漏标,错标等

数据扩充(数据增强)

简单的数据扩充方式

1.水平翻转:使原数据集扩充一倍

2.随机扣取:一般用较大的正方形(0.8-0.9的框进行抠图)再原图随机位置扣取图像,每张图像扣取的次数决定了数据集扩充的倍数

3.旋转:将原图宣传一定角度,将经旋转变换后的图像作为扩充的训练样本加入原训练集

一般(正负15度,30度,45度)

4.色彩抖动:在RGB颜色空间对原有RGB色彩分布进行轻微的扰动,或在HSV颜色控件随机改变原有图像饱和度和明度或对色调进行微调

在实践中,往往会将上述几种方式叠加使用,如果便可将图像数据扩充至原有数量的数倍甚至数十倍.

## 数据预处理

对输入特征做归一化处理预处理操作是常见步骤，同样在图像处理中

可以将图像的每个像素信息看作一种特征

在实践中，对每个特征减去平均值来中心化数据的归一化处理方式称为“中心式归一化”

需要注意的是，实际操作中应首先划分好训练集，验证集和测试集，该均值仅针对划分后的训练集计算，不可再未划分的所有图上计算，如此会违背机器学习的基本原理，即“在模型训练过程中能且仅能从训练集中获取信息”

### 样本不均衡的弊端

经典算法通常假设各类数据是均衡的，即每个类别的样本数量是一样多的

然而，实际工程多数情况样本是不均衡的

不均衡导致泛化能力弱

- 模型算法重视样本多数多的类别
- 轻视样本少数的类别

处理方式：把数据数量拉平

- 直接复制类别较少的图片
- 数据扩充的方式

样本数量过多的情况下，有针对性地挑选图片进行训练

下采样不是丢弃图片，我们可以选择每一次都从数据集中抽取数量相等地图片，而不是丢失图片。

### 类别均衡

每个mini-batch先随机挑一个或者几个类别

对每个类别从文件列表中均衡的随机挑选若干图片

### 迁移学习

只要有可能，就要尽量使用迁移学习，因为确实有用

由于已经加载了预训练模型，所以学习率应该比较低 例如10<sup>-4</sup>

### 学习率

模型训练开始时的初始学习率不宜过大，可以考虑设置为0.01和0.001

如果刚开始的几个mini-batch损失值就爆炸了，那么说明学习过大，需要调小学习率

在训练过程中，学习率应随着轮数的增加而减小

常见的减缓机制有

分段衰减：由给定step数分段呈阶梯状衰减，每段内学习率相同

指数衰减，每次将当前学习率乘以给定的衰减率得到下一个学习率

余弦衰减：学习率随step数变化呈余弦函数

损失值爆炸：大幅减少学习率

损失值降低缓慢：稍微加大学习率

后劲不足：使用较小的学习率从头练

## 算法精度提高的基本方法

### 神经网络初始化策略

初始化给出的值太大，那么可能导致梯度爆炸或者一直震荡不能收敛

初始化值

激活的平均值应该为0

激活的方差应该在每一层保持不变

使用Xavier初始化（或其派生初始化方法）

### 网络初始化的方案

1.全零初始化

不能将w初始化为0 b可以初始化为0 w可以初始化为一个较小的值

2.随机初始化

随机参数服从高斯分布和均匀分布，系统自动进行初始化

### 梯度下降策略

SGD->SGDM->NAG->AdaGrad->RMSProp->Adam

### SGD

最速梯度下降

1.梯度下降策略关键点：步长和方向

会出现局部最小值的问题

### Adagrad

学习率是自适应的

如果某个维度的梯度一直很大，那么该维度的学习率就会变的越来越小

