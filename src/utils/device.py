"""设备检测工具 - 自动选择 MPS/CUDA/CPU"""
import torch


def get_device():
    """
    自动检测并返回最佳可用设备

    优先级: MPS > CUDA > CPU

    Returns:
        torch.device: 可用的设备对象
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"使用设备: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"使用设备: CPU")

    return device


def print_device_info():
    """打印详细的设备信息"""
    device = get_device()

    print(f"\n{'='*50}")
    print(f"设备信息:")
    print(f"{'='*50}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"当前设备: {device}")

    if device.type == 'cuda':
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print(f"MPS 后端可用: {torch.backends.mps.is_available()}")
        print(f"MPS 已构建: {torch.backends.mps.is_built()}")

    print(f"{'='*50}\n")

    return device
