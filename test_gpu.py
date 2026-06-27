import torch


def check_pytorch_gpu():
    # 1. 检查 NVIDIA CUDA
    if torch.cuda.is_available():
        print(f"✅ GPU可用 (NVIDIA CUDA)")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU名称: {torch.cuda.get_device_name(0)}")
        return True

    # 2. 检查 Apple Silicon (M1/M2/M3) 金属加速
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✅ GPU可用 (Apple Metal Performance Shaders)")
        return True

    # 3. 检查 AMD ROCm
    if hasattr(torch, 'hip') and torch.hip.is_available():
        print(f"✅ GPU可用 (AMD ROCm)")
        return True

    print("❌ PyTorch 未检测到可用的GPU，当前使用CPU")
    return False


# 执行检测
check_pytorch_gpu()
