Date:2026-07-14
Enviroment:
-macOS
-Apple Silicon arm64
-Miniforge
-Python 3.10
-ManiSkill installed successfully

Result:
-'import mani_skill' successed
-'gym.make("PushCube-v1",obs_mode="state")' failed

Error:
RuntimeError:vk::CreateInstanceUnique:ErrorIncompatibleDriver

Reason:
The local macOS Vulkan/rendering backend is incompatible with ManiSkill/SAPIEN

Decision:
Use Mac for coding,notes,and project organization
Use Colab or Linux/cloud GPU for ManiSkill simulation experiments

2.1数据操作
n维数组也称为张量（tensor）。深度学习框架中的张量类（在MXNet中为ndarray，在PyTorch和TensorFlow中为Tensor）都与Numpy中的ndarray类似。但深度学习框架又比Numpy的darray多一些重要功能：
首先，GPU很好地支持加速计算，而NumPy仅支持CPU计算；其次，张量类支持自动微分。这些功能使得张量类更适合深度学习
2.1.1 
首先，导入torch
import torch
张量表示一个由数值组成的数组，这个数组可能有多个维度。具有一个轴的张量对应数学上的向量（vector）；具有两个轴的张量对应数学上的矩阵（matrix）；具有两个轴以上的张量没有特殊的数学名称
可以使用arange创建一个行向量x。这个行向量包含以0开始的前12个整数，它们默认创建为整数。也可以指定创建类型为浮点数。张量中的每个值都称为张量的元素（element）。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算
x=torch.arange(12)
可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状
x.shape
如果只想知道张量中元素的总数:x.numel()
要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数。例如，可以把张量x从形状(12,)的行向量转换为形状为(3,4)的矩阵。这个新的张量包含与转换前相同的值，但是它被看成一个3行4列的矩阵。虽然张量的形状发生了改变，但其元素值并没有变。注意，通过改变张量的形状，张量的大小不会改变
X=x.reshape(3,4)
我们不需要通过手动指定每个维度来改变形状。也就是说，如果我们的目标形状是（高度，宽度），那么在知道宽度后，高度会被自动计算得出，不必我们自己做除法。可以通过-1来调用此自动计算出维度的功能。可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)
有时，我们希望使用全0、全1、其他常量，或者从特定分布中随机采样的数字来初始化矩阵。我们可以创建一个形状为（2，3，4）的张量，其中所有元素都设置为0：torch.zeros((2,3,4))
所有元素都设置为1：torch.ones((2,3,4))
有时我们想通过从某个特定的概率分布中随机采样来得到张量中每个元素的值。例如，当我们构造数组来作为神经网络中的参数时，我们通常会随机初始化参数的值。以下代码创建一个形状为（3，4）的张量。其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
torch.randn(3,4)
我们还可以通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。在这里，最外层的列表对应于轴0，内层的列表对应于轴1
torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
2.1.2运算符
对于任意具有相同形状的张量，常见的标准算术运算符(+、-、*、/和**）都可以被升级为按元素运算。我们可以在同一形状的任意两个张量上调用按元素操作
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
x+y,x-y,x*y,x/y,x**y//**运算符是求幂运算
“按元素”方式可以应用更多的计算，包括像求幂这样的一元运算符
torch.exp(x)
除了按元素计算外，我们还可以执行线性代数运算，包括向量点积和矩阵乘法
我们也可以把多个张量连结（concatenate）在一起，把它们端到端地叠起来形成一个更大的张量。我们只需要提供张量列表，并给出沿哪个轴连结。下面的例子分别演示了当我们沿行（轴-0，形状的第一个元素）和按列（轴-1，形状的第二个元素）连结两个矩阵时，会发生什么情况，我们可以看到，第一个输出张量的轴-0长度(6)是两个输入张量轴-0长度的总和(3+3)；第二个输出张量的轴-1长度(8)是两个输入张量轴-1长度的总和(4+4)
X=torch.arange(12,dtype=torch.float32).reshape((3,4))
Y=torch.tensor([[2.0,1,,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1)
有时，我们想通过逻辑运算符构建二元张量。以X==Y为例：对于每个位置，如果X和Y在该位置相等，则新张量相应项的值为1。这意味着逻辑语句X==Y在该位置处为真，否则该位置为0
X==Y
对张量中的所有元素进行求和，会产生一个单元素张量
X.sum()
2.1.3广播机制
在上面的部分中，我们看到了如何在相同形状的两个张量上执行按元素操作。在某些情况下，即使形状不同，我们仍然可以通过调用广播机制(broadcasting mechanism)来执行按元素操作。这种机制的工作方式如下：
  1.通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状
  2.对生成的数组执行按元素操作
在大多数情况下，我们将沿着数组中长度为1的轴进行广播，如下例子：
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
由于a和b分别是3*1和1*2的矩阵，如果让它们相加，它们的形状不匹配。我们将矩阵广播为一个更大的3*2矩阵。如下所示，矩阵a复制列，矩阵b复制行，然后再按元素相加
a+b
2.1.4索引和切片
就像在任何其他Python数组中一样，张量中的元素可以通过索引访问。与任何Python数组一样：第一个元素的索引是0，最后一个元素的索引是-1；可以指定范围以包含第一个元素和最后一个之前的元素
如下所示，我们可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素：
X[-1],X[1:3]
除读取外，我们还可以通过指定索引来将元素写入矩阵：
X[1,2]=9
如果我们想为多个元素赋值相同的值，我们只需索引所有元素，然后为它们赋值。例如，[0:2,:]访问第1行和第2行，其中":"表示沿轴1（列）的所有元素。虽然我们讨论的是矩阵的索引，但这也适用于向量和超过2个维度的张量
X[0:2,:]=12
2.1.5节省内存
运行一些操作可能会导致为新结果分配内存。例如，如果我们用Y=X+Y，我们将取消引用Y指向的张量，而是指向新分配的内存处的张量
在下面的例子中，我们用Python的id()函数演示了这一点，它给我们提供了内存中引用对象的确切地址。运行Y=Y+X后，我们会发现id(Y)指向另一个位置。这是因为Python首先计算Y+X，为结果分配新的内存，然后使Y指向内存中的这个新位置
before=id(Y)
Y=Y+X
id(Y)==before
这可能是不可取的，原因有两个：
  1、首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新
  2、如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数

执行原地操作非常简单。我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:]=<expression>。
Z=torch.zeros_like(Y)
print('id(Z)',id(Z))
Z[:]=X+Y
print('id(Z)',id(Z))
如果在后续计算中没有重复使用X，我们也可以使用X[:]=X+Y或X+=Y来减少操作的内存开销
before=id(X)
X+=Y
id(X)==before
2.1.6转换为其他Python对象
将深度学习框架定义的张量转换为Numpy张量（ndarray）很容易，反之也同样容易。torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量
A=X.numpy()
B=torch.tensor(A)
type(A),type(B)
要将大小为1的张量转换为Python标量，，我们可以调用item函数或Python的内置函数
a=torch.tensor([3.5])
a,a.item(),float(a),int(a)

广播机制规则：
  如果遵守以下规则，则两个tensor是“可广播”的：
    每个tensor至少有一个维度
    遍历tensor所有维度时，从末尾开始遍历，两个tensor存在下列情况：
      tensor维度相等
      tensor维度不等且其中一个维度为1
      tensor维度不等且其中一个维度不存在
  如果两个tensor是”可广播“的，则计算过程遵循下列规则：
    如果两个tensor的维度不同，则在维度较小的tensor的前面增加维度，使它们维度相等
    对于每个维度，计算结果的维度值取两个tensor中较大的那个值
    两个tensor扩展维度的过程是将数值进行复制


2.2数据预处理
2.2.1读取数据集
我们首先创建一个人工数据集，并存储在CSV（逗号分隔值）文件../data/house_tiny.csv中。以其他格式存储的数据也可以通过类似的方式进行处理。下面我们将数据集按行写入CSV文件中
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')#列名
    f.write('NA,Pave,127500\n')#每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
要从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数。该数据集有四行三列。其中每行描述了房间数量("NumRooms")、巷子类型("Alley")和房屋价格("Price")
#如果没有安装pandas，只需取消对以下行的注释来安装pandas
#!pip install pandas
import pandas as pd
data=pd.read_csv(data_file)
print(data)
2.2.2处理缺失值
注意，“NAN”项表示缺失值。为了处理缺失的数据，典型的方法包括插值法和删除值，其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。在这里，我们将考虑插值法。
通过位置索引iloc，我们将data分成inputs和outputs，其中前者为data的前两列，而后者为data的最后一列。对于inputs中缺少的数值，我们用同一列的均值替换“NAN”项
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
inputs=inputs.fillna(inputs.mean())
print(inputs)
对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。由于“巷子类型”（”Alley“）列只接受两种类型的类别值”Pave“和”NaN“，pandas可以自动将此列转换为两列”Alley_Pave"和"Alley_nan"。巷子类型为“Pave”的行会将“Alley_Pave“的值设置为1，“Alley_nan“的值设置为0。缺少巷子类型的行会将"Alley_Pave"和"Alley_nan“分别设置为0和1
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)
2.2.3转换为张量格式
现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
import torch
X=torch.tensor(inputs.to_numpy(dtype=float))
y=torch.tensor(outputs.to_numpy(dtype=float))
X,y

2.3线性代数
2.3.1标量
仅包含一个数值被称为标量（scalar）。变量（variable），它们表示未知的标量值
标量由只有一个元素的张量表示。下面的代码将实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数
x=torch.tensor(3.0)
y=torch.tensor(2.0)
x+y,x*y,x/y,x**y
2.3.2向量
向量可以被视为标量值组成的列表。这些标量值被称为向量的元素（element）或分量（component）。当向量表示数据集中的样本时，它们的值具有一定的现实意义
人们通过一维张量表示向量。一般来说，张量可以具有任意长度，取决于机器的内存限制
x=torch.arange(4)
x
我们可以使用下标来引用向量的任一元素
x[3]
2.3.2.1长度、维度和形状
向量只是一个数字数组，向量的长度通常称为向量的维度（dimension）
与普通的Python数组一样，我们可以通过调用Python的内置len()函数来访问张量的长度
len(x)
当用张量表示一个向量（只有一个轴）时，我们也可以通过.shape属性访问向量的长度。形状(shape)是一个元素组，列出了张量沿每个轴的长度（维度）。对于只有一个轴的张量，形状只有一个元素
x.shape
向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。然而，张量的维度用来表示张量具有的轴数。在这个意义上，张量的某个轴的维数就是这个轴的长度
2.3.3矩阵
正如向量将标量从零阶推广到一阶，矩阵将向量丛一阶推广到二阶。在代码中表示为具有两个轴的张量
当矩阵具有相同数量的行和列时，其形状将变为正方形；因此，它被称为方阵（square matrix）
当调用函数来实例化张量时，我们可以通过指定两个分量m和n来创建一个形状为m*n的矩阵
A=torch.arange(20).reshape(5,4)
A
当我们交换矩阵的行和列时，结果称为矩阵的装置（transpose）。
A.T
作为方阵的一种特殊类型，对称矩阵（symmetric matrix）A等于其装置

2.3.4张量
就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的n维数组的通用方法。例如，向量是一阶张量，矩阵是二阶张量。
当我们开始处理图像时，张量将变得更加重要，图像以n维数组形式出现，其中3个轴对应于高度、宽度，以及一个通道（channel）轴，用于表示颜色通道（红色、绿色和蓝色）。
X=torch.arange(24).reshape(2,3,4)
X

2.3.5张量算法的基本性质
标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。例如，从按元素操作的定义中可以注意到，任何按元素的一元运算都不会改变其操作数的形状。同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法
A=torch.arange(20,dtype=torch.dloat32).reshape(5,4)
B=A.clone()#通过分配新内存，将A的一个副本分配给B
A，A+B
具体而言，两个矩阵的按元素乘法称为Hadamard积（hadamard product）
A*B
将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘
a=2
X=torch.arange(24).reshape(2,3,4)
a+X,(a*X).shape

2.3.6降维
我们可以对任意张量进行的一个有用操作是计算其元素的和。
x=torch.arange(4,dtype=torch.float32)
x,x.sum()
默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。我们还可以指定张量沿哪一个轴来通过求和降低维度。以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定axis=0。由于输入矩阵沿0轴降维一生成输出向量，因此输入轴0的维数在输出形状中消失
A_sum_axis0=A.sum(axis=0)
A_sum_axis0,A_sum_axis0.shape
指定axis=1将通过汇总所有列的元素降维（轴1）。因此，输出轴1的维数在输出形状中消失
A_sum_axis1=A.sum(axis=1)
A_sum_axis1,A_sum_axis1.shape
沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和
A.sum(axis=[0,1])
一个与求和相关的量是平均值（mean或average）。我们通过将总和除以元素总数来计算平均值。在代码中，我们可以调用函数来计算任意形状张量的平均值
A.mean(),A.sum()/A.numel()
同样，计算平均值的函数也可以沿指定轴降低张量的维度
A.mean(axis=0),A.sum(axis=0)/A.shape[0]
2.3.6.1非降维求和
但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用
sum_A=A.sum(axis=1,keepdims=True)
sum_A
例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A
A/sum_A
如果我们想沿某个轴计算A元素的累积总和，比如axis=0（按行计算），可以调用cumsum函数。此函数不会沿任何轴降低输入张量的维度
A.cumsum(axis=0)
2.3.7点积（Dot Product）
点积是相同位置的按元素乘积的和
y=torch.ones(4,dtype=torch.float32)
x,y.torch.dot(x,y)
我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
torch.sum(x*y)
2.3.8矩阵-向量积
矩阵-向量积（matrix-vector priduct）
在代码中使用张量表示矩阵-向量积，我们使用mv函数。当我们为矩阵A和向量x调用torch.mv(A,x)时，会执行矩阵-向量积。注意，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同
A.shape,x.shape,torch.mv(A,x)
2.3.9矩阵-矩阵乘法（matrix-matrix multiplication）
B=torch.ones(4,3)
torch.mm(A,B)
矩阵-矩阵乘法可以简单地称为矩阵乘法

2.3.10范数（norm）
非正式地说，向量的范数是表示一个向量有多大。这里考虑的大小（size）概念不涉及维度，而是分量的大小
在线形代数中，向量范数是将向量映射到标量的函数f。 
欧几里得距离是一个L2范数，L2范数是向量元素平方和的平方根
可以按如下方式计算向量的L2范数
u=torch.tensor([3.0,-4.0])
torch.norm(u)
深度学习中更经常地使用L2范数的平方，也会经常遇到L1范数，它表示为向量元素的绝对值之和
与L2范数相比，L1范数受异常值的影响较小。为了计算L1范数，我们将绝对值函数和按元素求和组合起来
torch.abs(u).sum()
矩阵的Frobenius范数（Frobenius norm）是矩阵元素平方和的平方根
Frobenius范数满足向量范数的所有性质，它就像是矩阵向量的L2范数。调用以下函数将计算矩阵的Frobenius范数
torch.norm(torch.ones((4,9)))
2.3.10.1范数和目标
在深度学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率；最小化预测和真实观测之间的距离。用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数

2.4微积分
在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。通常情况下，变得更好意味着最小化一个损失函数（lossfunction），即一个衡量“模型有多糟糕”这个问题的分数。
最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。但“训练”模型只能将模型与我们实际能看到的数据相拟合。因此，我们可以将拟合模型的任务分解为两个关键问题：
  优化（optimization）：用模型拟合观测数据的过程
  泛化（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型
2.4.1导数和微分
在深度学习中，我们通常选择对于模型参数可微的损失函数。简而言之，对于每个参数，如果我们把这个参数增加或减少一个无穷小的量，可以知道损失以多块的速度增加或减少
%matplotlib inline
import numpy as np
from matplotlib_inline import backend_inline
form d2l import torch as d2l

def f(x):
  return 3*x**2-4*x
def numerical_lim(f,x,h):
  return (f(x+h)-f(x))/h
h=0.1
for i in range(5):
  print(f'h={h:.5f},numerical limit={numerical_lim(f,1,h):.5f}')
  h*=0.1
为了对导数的这种解释进行可视化，我们将使用matplotlib，这是一个Python中流行的绘图库。要配置matplotlib生成图形的属性，我们需要定义几个函数。在下面，use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像
注意，注释#@save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。因此，以后无须重新定义就可以直接调用它们（例如，d2l.use_svg_display()）
def use_svg_display():#@save
  """设置matplotlib的图表大小"""
  use_svg_display()
  d2l.plt.rcParams['figure.figsize']=figsize
下面的set_axes函数用于设置由matplotlib生成图表的轴的属性
#@save
def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
  """设置matplotlib的轴"""
  axes.set_xlabel(xlabel)
  axes.set_ylabel(ylabel)
  axes.set_xscale(xscale)
  axes.set_yscale(yscale)
  axes.set_xlim(xlim)
  axes.set_ylim(ylim)
  if legend:
    axes.legend(legend)
  axes.grid()
通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线
#@save
def plot(X,Y=None,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',fmts=('-'.'m--','g--','r:'),figsize=(3.5,2.5),axes=None):
  ""'绘制数据点"""
  if legend is None:
    legend=[]
  set_figsize(figsize)
  axes=axes if axes else d2l.plt.gca()
  #如果X有一个轴，输出True
  def has_one_axis(X):
    return (hasattr(X,"ndim") and X.ndim==1 or isinstance(X,list) and not hasattr(X[0],"__len__"))
  if has_one_axis(X):
    X=[X]
  if Y is None:
    X,Y=[[]]*len(X),X
  elif has_one_axis(Y):
    Y=[Y]
  if len(X)!=len(Y):
    X=X*len(Y)
  axes.cla()
  for x,y fmt in zip(X,Y,fmts):
    if len(x):
      axes.plot(x,y,fmt)
    else:
      axes.plot(y,fmt)
  set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)

x=np.arange(0,3,0.1)
plot(x,[f(x),2*x-3],'x','f(x)',legend=['f(x)','Tangent line(x=1)'])
2.4.2偏导数
2.4.3梯度
微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题
导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率
梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数
链式法则可以用来微分复合函数
<img width="697" height="127" alt="image" src="https://github.com/user-attachments/assets/0f56dc2b-28a1-4a2c-9f16-0037367c9bfb" />

2.5自动微分
深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。实际上，根据设计好的模型，系统会构建一个计算图（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数
2.5.1一个简单的例子
作为一个演示例子，假设我们想对函数y=2X^TX关于列向量X求导。首先我们创建变量X并为其分配一个初始值
import torch
x=torch.arange(4.0)
x
在我们计算y关于X的梯度之前，需要一个地方来存储梯度。重要的是，我们不会在每次对一个参数求导时都分配新的内存。因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能会快就会将内存耗尽。注意，一个标量函数关于向量X的梯度是向量，并且与X具有相同的形状
x.requries_grad(True)#等价于x=torch.arange(4.0,requires_grad=True)
x.grad#默认值是None

现在计算y
y=2*torch.dot(x,x)
y
x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。接下来，通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度
y.backward()
x.grad
函数y=2X^X关于X的梯度应为4X。让我们快速验证这个梯度是否计算正确
x.grad==4*x
现在计算x另一个函数
#在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y=x.sum()
y.backward()
x.grad
2.5.2非标量变量的反向传播
当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。对于高阶和高纬的y和x。求导的结果可以是一个高阶张量
当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和
#对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度
#本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y=x*x
#等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
2.5.3分离计算
有时，我们希望将某些计算移动到记录的计算图之外。例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数，并且只考虑到x在y被计算后发挥的作用
这里可以分离y来返回一个新变量u，该变量与y具有相同的值，但丢弃计算图中如何计算y的任何信息。换句话说，梯度不会向后流经u到x。因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理，而不是z=x*x*x关于x的偏导数
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
z.sum().backward()
x.grad==u
由于记录了y的计算结果，我们随后在y上调用反向传播，得到y=x*x关于x的导数，即2*x
x.grad.zero_()
y.sum().backward()
x.grad==2*x
2.5.4Python控制流的梯度计算
使用自动微分的一个好处是：即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到变量的梯度。在下面的代码中，while循环的迭代次数和if语句的结果都取决于输入a的值
def f(a):
  b=a*2
  while b.norm()<1000:
    b=b*2
  if b.sum()>0:
    c=b
  else:
    c=100*b
  return c

让我们计算梯度
a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()
我们现在可以分析上面定义的f函数。它在其输入a中是分段线性的。换言之，对于任何a，存在某个常量标量k，使得f(a)=k*a，其中k的值取决于输入a，因此可以用d/a验证梯度是否正确
a.grad==d/a
<img width="701" height="210" alt="image" src="https://github.com/user-attachments/assets/4f5db5ae-9e06-4404-aac0-e3838923ae43" />

2.6概率
2.6.1基本概率论
对于每个值，一种自然的方法是将它出现的次数除以投掷的总次数，即此事件（event）概率的估计值。
大数定律（law of large numbers）告诉我们：随着投掷次数的增加，这个估计值会越来越接近真实的潜在概率。
%matplotlib inline
import torch
from torch.distributions import multinomial
from d2l import torch as d2l
在统计学中，我们把从概率分布中抽取样本的过程称为抽样（sampling）。笼统来说，可以把分布（distribution）看作对事件的概率分配。将概率分配给一些离散选择的分布称为多项分布（multinomial distribution）
为了抽取一个样本，即投骰子，我们只需传入一个概率向量。输出是另一个相同长度的向量：它在索引i处的值是采样结果中i出现的次数
fair_probs=torch.ones([6])/6
multinomial.Multinomial(1,fair_probs).sample()
在估计一个骰子的公平性时，我们希望从同一分布中生成多个样本。如果用Python的for循环来完成这个任务，速度会很慢。因此我们使用深度学习框架的函数同时抽取多个样本，得到我们想要的任意形状的独立样本数组
multinomial.Multinomial(10,fair_probs).sample()
我们可以模拟1000次投掷。然后我们可以统计1000次投掷后，每个数字被投中了多少次。具体来说，我们计算相对频率，以作为真实概率的估计
#将结果存储为32位浮点数以进行除法
counts=multinomial.Multinomial(1000,fair_probs).sample()
counts/1000#相对频率作为估计值
我们也可以看到这些概率如何随着时间的推移收敛到真实概率。让我们进行500组实验，每组抽取10个样本
counts=multinomial.Multinomial(10,fair_probs).sample((500,))
cum_counts=counts.cumsum(dim=0)
estimates=cum_counts/cum_counts.sum(dim=1,keepdims=True)
d2l.set_figsize((6,4.5))
for i in range(6):
  d2l.plt.plot(estimates[:,i].nnumpy(),label=("P(die="+str(i+1)+")"))
d2l.plt.axhline=(y=0.167,color='black',linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();




























