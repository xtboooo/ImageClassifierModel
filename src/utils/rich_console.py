"""Rich 终端美化工具模块"""
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from typing import List, Dict, Any, Optional
import io
from pathlib import Path


# 全局 Console 实例(单例模式)
console = Console()

# 用于记录日志的文件对象
_log_file = None


def setup_rich_logging(log_dir: Path):
    """
    设置 Rich 输出同时保存到日志文件

    Args:
        log_dir: 日志目录路径
    """
    global _log_file
    if _log_file is not None:
        _log_file.close()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 打开日志文件用于追加
    _log_file = open(log_dir / "rich_output.log", "a", encoding="utf-8")


def _log_to_file(content: str):
    """
    将内容写入日志文件（纯文本格式）

    Args:
        content: 要记录的内容
    """
    if _log_file is not None:
        # 使用 Console 渲染为纯文本
        string_io = io.StringIO()
        temp_console = Console(file=string_io, force_terminal=False, width=120)
        temp_console.print(content)
        _log_file.write(string_io.getvalue())
        _log_file.flush()


def setup_rich():
    """
    初始化 rich 配置

    注意: Rich 的默认配置已经足够好,此处预留扩展接口
    """
    pass


class RichProgressManager:
    """
    Rich 进度条管理器

    支持多任务并行显示,替代 tqdm。

    Features:
    - 自动旋转图标(SpinnerColumn)
    - 进度条(BarColumn)
    - 百分比(TaskProgressColumn)
    - 剩余时间(TimeRemainingColumn)
    - 完成后保留显示(transient=False)

    Example:
        >>> with RichProgressManager() as progress:
        ...     task1 = progress.add_task("训练", total=100)
        ...     task2 = progress.add_task("验证", total=50)
        ...     for i in range(100):
        ...         train()
        ...         progress.update("训练", advance=1)
        ...     for i in range(50):
        ...         validate()
        ...         progress.update("验证", advance=1)
    """

    def __init__(self, transient: bool = False):
        """
        初始化进度条管理器

        Args:
            transient: 完成后是否清除进度条,默认 False(保留显示)
        """
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=transient
        )
        self.tasks = {}

    def __enter__(self):
        """上下文管理器入口"""
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        """上下文管理器出口"""
        self.progress.__exit__(*args)

    def add_task(self, name: str, total: int, **kwargs):
        """
        添加任务

        Args:
            name: 任务名称
            total: 任务总量
            **kwargs: 传递给 Progress.add_task 的其他参数

        Returns:
            task_id: Rich Progress 任务 ID
        """
        task_id = self.progress.add_task(name, total=total, **kwargs)
        self.tasks[name] = task_id
        return task_id

    def update(self, name: str, advance: int = 1, **kwargs):
        """
        更新任务进度

        Args:
            name: 任务名称
            advance: 前进步数,默认 1
            **kwargs: 传递给 Progress.update 的其他参数(如 description)
        """
        if name in self.tasks:
            self.progress.update(self.tasks[name], advance=advance, **kwargs)

    def remove_task(self, name: str):
        """
        移除任务

        Args:
            name: 任务名称
        """
        if name in self.tasks:
            self.progress.remove_task(self.tasks[name])
            del self.tasks[name]


# ========== 便捷函数 ==========


def print_header(title: str, subtitle: str = None):
    """
    打印美化的标题面板

    Args:
        title: 主标题
        subtitle: 副标题(可选)

    Example:
        >>> print_header("训练流水线", "完整的训练、评估、导出流程")
    """
    panel_content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        panel_content += f"\n[dim]{subtitle}[/dim]"

    panel = Panel(panel_content, expand=False)
    console.print(panel)
    _log_to_file(panel)


def print_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    caption: str = None
):
    """
    打印美化的表格

    Args:
        title: 表格标题
        headers: 表头列表
        rows: 行数据列表(每行是一个列表)
        caption: 表格底部说明文字(可选)

    Example:
        >>> print_table(
        ...     title="数据集统计",
        ...     headers=["数据集", "数量"],
        ...     rows=[["训练集", 1000], ["验证集", 200]],
        ...     caption="总计: 1200 张图片"
        ... )
    """
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        caption=caption
    )

    # 添加列
    for header in headers:
        table.add_column(header)

    # 添加行
    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)
    _log_to_file(table)


def print_panel(content: str, title: str = None, style: str = "green"):
    """
    打印面板

    Args:
        content: 面板内容(支持 Rich markup)
        title: 面板标题(可选)
        style: 面板样式(颜色),默认 green

    Example:
        >>> print_panel(
        ...     "[bold green]训练完成![/bold green]\\n最佳准确率: 95.2%",
        ...     title="训练总结",
        ...     style="green"
        ... )
    """
    panel = Panel(content, title=title, style=style)
    console.print(panel)
    _log_to_file(panel)


def print_syntax(code: str, language: str = "python", theme: str = "monokai", line_numbers: bool = True):
    """
    打印语法高亮代码

    Args:
        code: 代码字符串
        language: 语言类型(python, json, yaml 等)
        theme: 主题,默认 monokai
        line_numbers: 是否显示行号,默认 True

    Example:
        >>> config_str = '{"model": "mobilenet_v2", "epochs": 30}'
        >>> print_syntax(config_str, language="json")
    """
    syntax = Syntax(code, language, theme=theme, line_numbers=line_numbers)
    console.print(syntax)


def print_tree(root_name: str, structure: Dict):
    """
    打印文件树结构

    Args:
        root_name: 根节点名称
        structure: 树形结构字典(嵌套字典表示目录,其他值表示文件)

    Example:
        >>> structure = {
        ...     "logs": {
        ...         "main.log": None,
        ...         "errors.log": None
        ...     },
        ...     "checkpoints": {
        ...         "best_model.pth": None
        ...     }
        ... }
        >>> print_tree("运行目录", structure)
    """
    tree = Tree(f"[bold]{root_name}[/bold]")

    def add_nodes(parent, items):
        """递归添加节点"""
        for key, value in items.items():
            if isinstance(value, dict):
                branch = parent.add(f"[blue]{key}/[/blue]")
                add_nodes(branch, value)
            else:
                parent.add(f"[green]{key}[/green]")

    add_nodes(tree, structure)
    console.print(tree)


def print_info(message: str):
    """
    打印信息消息

    Args:
        message: 消息内容
    """
    console.print(f"[cyan]ℹ[/cyan] {message}")


def print_success(message: str):
    """
    打印成功消息

    Args:
        message: 消息内容
    """
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str):
    """
    打印警告消息

    Args:
        message: 消息内容
    """
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_error(message: str):
    """
    打印错误消息

    Args:
        message: 消息内容
    """
    console.print(f"[red]✗[/red] {message}")


def print_stage_header(stage_num: int, total_stages: int, title: str, description: str = None):
    """
    打印统一的阶段标题（使用表格形式）

    Args:
        stage_num: 当前阶段编号
        total_stages: 总阶段数
        title: 阶段标题
        description: 阶段描述（可选）

    Example:
        >>> print_stage_header(1, 8, "模型训练", "使用较小学习率微调整个模型")
    """
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        collapse_padding=True
    )
    table.add_column(style="bold cyan", width=15)
    table.add_column(style="white")

    table.add_row(f"[阶段 {stage_num}/{total_stages}]", f"[bold]{title}[/bold]")
    if description:
        table.add_row("", f"[dim]{description}[/dim]")

    console.print("")
    console.print(table)
    _log_to_file(table)


# 导出
__all__ = [
    'console',
    'setup_rich',
    'setup_rich_logging',
    'RichProgressManager',
    'print_header',
    'print_table',
    'print_panel',
    'print_syntax',
    'print_tree',
    'print_info',
    'print_success',
    'print_warning',
    'print_error',
    'print_stage_header',
]
