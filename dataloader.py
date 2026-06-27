from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size, test_batch_size, data_root='./data', use_accel=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': test_batch_size}
    if use_accel:
        # 加速器相关参数（根据实际情况调整）
        train_kwargs.update({'num_workers': 1, 'persistent_workers': True, 'pin_memory': True})
        test_kwargs.update({'num_workers': 1, 'persistent_workers': True, 'pin_memory': True})

    train_loader = DataLoader(train_dataset, **train_kwargs) # **是Python中的字典解包操作符,将字典展开作为函数参数
    test_loader = DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader