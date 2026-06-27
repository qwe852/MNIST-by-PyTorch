# 以MNIST为项目的Pytorch入门教程
## PyTorch的官方文档
PyTorch的官方文档是PyTorch学习者最重要的学习资源
网址为https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
## 本项目结构为：
```
mnist_pytorch/  
├── README.md  
├── config.py                # 超参数设置  
├── dataloader.py            # 数据加载与预处理  
├── models.py                # 神经网络定义  
├── engine/  
│   ├── trainer.py           # 训练循环  
│   └── tester.py            # 测试评估  
├── main.py                  # 程序入口（解析参数，协调各模块）  
├── note/  
│   └── notebook.py          # 学习笔记  
│   └── notebook.ipynb       # 学习笔记  
├── test_gpu.py              # 测试cuda环境
```
## tensor
### tensor和numpy的区别和联系
PyTorch的Tensor和NumPy的ndarray都是用来存储和操作多维数据的
但是numpy只能在cpu上运行，而tensor可以放在gpu上并且支持自动求导、gpu加速和神经网络集成<br>
他们可以通过.numpy()和from_numpy相互转换
### tensor的基础用法
torch.tensor()  
torch.zeros()  
torch.ones()  
torch.randn()  
arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor  
按照步长依次生成[start,end)范围内的数列
```text
    torch.arange(5)  # 默认以 0 为起点
    tensor([ 0,  1,  2,  3,  4])
    torch.arange(1, 4)  # 默认间隔为 1
    tensor([ 1,  2,  3])
    torch.arange(1, 2.5, 0.5)  # 指定间隔 0.5
    tensor([ 1.0000,  1.5000,  2.0000])
```
## 神经网络组成
### nn.Conv2d
定义一个卷积核，方便后面进行二维卷积操作  
参数分别可以规定图片输入的尺寸，输出的尺寸，卷积核大小，步长等参数 

    class torch.nn.Conv2d(
                         in_channels, 
					     out_channels,
				         kernel_size,
						 stride=1, 
						 padding=0, 
						 dilation=1, 
						 groups=1, 
						 bias=True)

卷积输出尺寸大小公式：(W-K+2P)/S+1  
K = kernel_size, P = padding, S = stride

### nn.max_pool2d
对由多个输入平面组成的输入信号应用二维最大池化，对于每个卷积核内的所有值取最大值，作用  
减少计算量  
减少噪声——消除最小较小的值  
扩大感受野——网络能够看到更大的区域

    class torch.nn.max_pool2d(
                              kernel_size
                              stride
                              padding
                              dilation
                              return_indices
                              ceil_mode )

## 优化器
优化器是一种特定的深度学习算法，负责调整神经网络的权重和偏置，以便最小化损失函数，从而提高模型的准确性和性能  
常见的优化器，包括梯度下降系列（批量梯度下降BGD、随机梯度下降SGD、
小批量梯度下降MBGD）、动量法、NAG、Adagrad、RMSprop以及Adam等，
它们的核心目标是通过调整学习率、利用梯度信息等手段，高效地最小化损失函数，
从而优化和提升神经网络模型的性能  
### 随机梯度下降（SGD）
每次迭代仅使用一个训练样本来计算损失函数的梯度，并更新模型参数  
适用于大规模数据集和在线学习场景  

### 动量法（Momentum）
通过引入一个累计梯度的指数加权平均，将过去的梯度信息考虑进当前的参数更新中，从而增加稳定性和提高训练效率。  
常用于改进随机梯度下降（SGD）和小批量梯度下降（MBGD）等优化器。

### NAG
在动量法基础上进行改进的优化算法，先按照之前的动量更新参数，再在这个新的位置计算梯度，并根据此调整更新方向

### Adagrad
一种自适应梯度下降的优化器，对不同参数使用不同的学习率。对于更新频率较低的参数施以较大的学习率，对于更新频率较高的参数使用较小的学习率。  
适用于大规模数据集和特征提取任务

### RMSprop
对Adagrad的一种改进，根据梯度的历史信息来自适应地调整学习率。使用梯度的指数加权平均而不是累积和来计算学习率  
适用于处理非稀疏数据和长期依赖的问题

### Adadelta
对Adagrad的另一种改进，通过计算梯度平方的指数加权移动平均来避免学习率趋于0的问题，同时简化了计算。  
适用于需要长时间训练的大型神经网络和需要稳定学习率的任务。

### Adam
结合了AdaGrad和Momentum两种优化算法的优点，能够快速收敛并且减少训练时间。Adam优化器计算出每个参数的独立自适应学习率，不需要手动调整学习率的大小。  
适用于处理大规模数据和训练复杂模型

