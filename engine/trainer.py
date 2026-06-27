import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
def train(args, model, device, train_loader, optimizer, epoch, writer=None, global_step=0):
    model.train() # 设置model为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 清空梯度
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() # 按照计算的梯度执行梯度下降

        global_step+=1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            if args.dry_run:
                break