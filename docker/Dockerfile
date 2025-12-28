# ============================================
# Stage 1: Builder - 安装所有依赖
# ============================================
FROM python:3.10-slim-bullseye AS builder

# 安装 PyTorch CPU 版本（转换不需要 GPU）
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 安装 ai-edge-torch 和必要依赖（使用最新版本避免依赖冲突）
RUN pip install --no-cache-dir \
    ai-edge-torch \
    pillow \
    numpy \
    loguru

# ============================================
# Stage 2: Runtime - 最小化运行时镜像
# ============================================
FROM python:3.10-slim-bullseye

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 设置工作目录
WORKDIR /workspace

# 添加标签
LABEL maintainer="ImageClassifierModel"
LABEL description="TFLite model export using ai-edge-torch"
LABEL version="1.0"

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import ai_edge_torch; print('OK')" || exit 1

# 默认命令
CMD ["python", "--version"]

