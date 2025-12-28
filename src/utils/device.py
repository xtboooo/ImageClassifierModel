"""设备检测工具 - 自动选择 MPS/CUDA/CPU"""
import torch
from .logger import logger
from .rich_console import print_panel


def get_device():
    """
    自动检测并返回最佳可用设备

    优先级: MPS > CUDA > CPU

    Returns:
        torch.device: 可用的设备对象
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("使用设备: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("使用设备: CPU")

    return device


def print_device_info():
    """打印详细的设备信息"""
    device = get_device()

    # 构建设备信息内容
    info_lines = [
        f"[bold]PyTorch 版本[/bold]: {torch.__version__}",
        f"[bold]当前设备[/bold]: {device.type.upper()}"
    ]

    if device.type == 'cuda':
        info_lines.extend([
            f"[bold]CUDA 版本[/bold]: {torch.version.cuda}",
            f"[bold]GPU 数量[/bold]: {torch.cuda.device_count()}",
            f"[bold]当前 GPU[/bold]: {torch.cuda.current_device()}",
            f"[bold]GPU 名称[/bold]: {torch.cuda.get_device_name(0)}"
        ])
        logger.info("设备信息",
                    pytorch_version=torch.__version__,
                    device="CUDA",
                    cuda_version=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(0))
    elif device.type == 'mps':
        info_lines.extend([
            f"[bold]MPS 后端可用[/bold]: {torch.backends.mps.is_available()}",
            f"[bold]MPS 已构建[/bold]: {torch.backends.mps.is_built()}"
        ])
        logger.info("设备信息",
                    pytorch_version=torch.__version__,
                    device="MPS")
    else:
        logger.info("设备信息",
                    pytorch_version=torch.__version__,
                    device="CPU")

    print_panel("\n".join(info_lines), title="设备信息", style="cyan")

    return device
