import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import Net
from dataloader import get_data_loaders
from engine.trainer import train
from engine.tester import test
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
import atexit
import os
import sys

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--tb-port', type=int, default=6006, help='TensorBoard port')
    parser.add_argument('--no-tb', action='store_true', help='Do not launch TensorBoard')
    # 添加--tb-port参数和--no-tb，设置tensorboard的监听端口以及是否要开启tensorboard
    args = parser.parse_args()



    torch.manual_seed(args.seed) # 设置随机种子

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    # 设置设备
    # 如果未禁用，则启动 TensorBoard
    if not args.no_tb:
        # 确定 tensorboard 可执行文件位置
        # 优先使用当前 Python 解释器同级目录下的 tensorboard 脚本
        python_dir = os.path.dirname(sys.executable)  # 例如 ...\envs\gpupytorch\
        tb_path = os.path.join(python_dir, 'Scripts', 'tensorboard.exe')  # Windows
        if not os.path.exists(tb_path):
            # 如果不在 Scripts，尝试直接用 'tensorboard'（可能在 PATH 中）
            tb_path = 'tensorboard'

        cmd = [
            tb_path,
            '--logdir', args.log_dir,
            '--port', str(args.tb_port),
            '--bind_all'
        ]
        tb_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'TensorBoard launched at http://localhost:{args.tb_port}')
        atexit.register(tb_process.terminate)
        time.sleep(3)

    train_loader, test_loader = get_data_loaders(
        args.batch_size, args.test_batch_size, use_accel=use_accel
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,writer, global_step)
        test(model, device, test_loader,writer, epoch)
        scheduler.step()

    writer.close()# 关闭tensorboard

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()