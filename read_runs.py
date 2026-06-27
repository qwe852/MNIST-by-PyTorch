from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# 1. 指定事件文件所在的目录（runs 下具体的子文件夹）
log_dir = "./runs/events.out.tfevents.1782569424.LAPTOP-S431DIJC.12644.0"  # 换成你自己的文件夹名

# 2. 加载事件
event_acc = EventAccumulator(log_dir)
event_acc.Reload()  # 读取所有事件数据

# 3. 查看有哪些可用的标量标签
print("Available scalars:", event_acc.Tags()['scalars'])

# 4. 提取某个标量，例如训练损失 'Loss/train'
if 'Loss/train' in event_acc.Tags()['scalars']:
    loss_events = event_acc.Scalars('Loss/train')
    steps = [e.step for e in loss_events]
    values = [e.value for e in loss_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("'Loss/train' not found")

# 5. 也可以同时画出多个曲线
test_loss_events = event_acc.Scalars('Loss/test')
test_acc_events = event_acc.Scalars('Accuracy/test')

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot([e.step for e in test_loss_events], [e.value for e in test_loss_events], 'b-', label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')

ax2 = ax1.twinx()
ax2.plot([e.step for e in test_acc_events], [e.value for e in test_acc_events], 'r-', label='Test Accuracy')
ax2.set_ylabel('Accuracy (%)', color='r')

fig.legend(loc='upper right')
plt.title('Test Metrics')
plt.show()