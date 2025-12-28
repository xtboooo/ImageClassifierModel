"""日志工具模块 - 基于 loguru"""
from loguru import logger
from pathlib import Path
from datetime import datetime
import sys


class LogConfig:
    """日志配置类"""

    def __init__(
        self,
        run_dir: Path = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """
        初始化日志配置

        Args:
            run_dir: 运行目录,日志将保存在其 logs/ 子目录下
            console_level: 控制台日志级别
            file_level: 文件日志级别
        """
        if run_dir is None:
            run_dir = Path("data/output/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / "logs"
        self.console_level = console_level
        self.file_level = file_level


def setup_logger(
    run_dir: Path = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG"
):
    """
    配置 loguru logger

    实现双输出机制:
    1. 控制台: 简洁格式,彩色输出,默认 INFO 级别
    2. 文件: 详细格式,包含 3 种日志文件:
       - main.log: 所有日志(纯文本)
       - main.json: 结构化日志(JSON 格式)
       - errors.log: 仅错误和警告

    特性:
    - 异步写入(enqueue=True)避免阻塞训练
    - 自动轮转(50MB)和压缩
    - 保留 30 天

    Args:
        run_dir: 运行目录(会创建 logs/ 子目录),默认为 data/output/runs/YYYYMMDD_HHMMSS
        console_level: 控制台日志级别,默认 INFO
        file_level: 文件日志级别,默认 DEBUG

    Returns:
        logger: 配置好的 loguru logger 实例

    Example:
        >>> from src.utils.logger import setup_logger, logger
        >>> setup_logger(Path("data/output/runs/20251227_120000"))
        >>> logger.info("训练开始", epoch=1, total=30)
        >>> logger.success("模型保存成功", path="model.pth")
    """
    config = LogConfig(run_dir, console_level, file_level)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # 移除默认 handler
    logger.remove()

    # 1. 控制台输出 - 简洁格式,彩色
    logger.add(
        sys.stdout,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        enqueue=False  # 控制台输出不需要异步
    )

    # 2. 主日志文件 - 详细格式,纯文本
    logger.add(
        config.log_dir / "main.log",
        level=file_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",  # 50MB 轮转
        retention="30 days",  # 保留 30 天
        compression="gz",  # 压缩旧日志
        encoding="utf-8",
        enqueue=True  # 异步写入,提高性能
    )

    # 3. JSON 日志文件 - 结构化,便于分析
    logger.add(
        config.log_dir / "main.json",
        level=file_level,
        format="{message}",
        serialize=True,  # JSON 序列化
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True
    )

    # 4. 错误日志文件 - 仅 WARNING 及以上
    logger.add(
        config.log_dir / "errors.log",
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        rotation="10 MB",
        retention="60 days",  # 错误日志保留更久
        encoding="utf-8",
        enqueue=True
    )

    logger.info("日志系统已初始化", run_dir=str(config.run_dir), log_dir=str(config.log_dir))

    # 初始化 Rich 日志记录
    try:
        from src.utils.rich_console import setup_rich_logging
        setup_rich_logging(config.log_dir)
        logger.debug("Rich 日志记录已启用")
    except Exception as e:
        logger.warning(f"Rich 日志记录初始化失败: {e}")

    return logger


# 导出全局 logger 实例,其他模块可以直接导入使用
__all__ = ['logger', 'setup_logger']
