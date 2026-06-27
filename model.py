import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) # 提取低级特征，维度32 26 26
        x = self.conv2(x)
        x = F.relu(x) # 提取高一级特征，维度64 24 24
        x = F.max_pool2d(x, 2) # 池化，保留关键信息的情况下降低分辨率，维度64 12 12
        x = self.dropout1(x) #防止过拟合
        x = torch.flatten(x, 1) #三维展开为1维向量,全连接层只能输入一维向量
        x = self.fc1(x)
        x = F.relu(x) # 融合全局特征
        x = self.dropout2(x)
        x = self.fc2(x) #计算logits
        output = F.log_softmax(x, dim=1)
        return output