# 以MNIST为项目的Pytorch入门教程
## PyTorch的官方文档
PyTorch的官方文档是PyTorch学习者最重要的学习资源
网址为https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

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
```python
>>> torch.arange(5)  # 默认以 0 为起点
    tensor([ 0,  1,  2,  3,  4])
    torch.arange(1, 4)  # 默认间隔为 1
    tensor([ 1,  2,  3])
    torch.arange(1, 2.5, 0.5)  # 指定间隔 0.5
    tensor([ 1.0000,  1.5000,  2.0000])

