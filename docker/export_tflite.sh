#!/bin/bash
# ============================================
# TFLite 模型导出脚本（使用 Docker）
# 用于在 macOS/Windows 上导出 TFLite 模型
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取项目根目录（脚本所在目录的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 参数默认值
CHECKPOINT="${1:-data/output/checkpoints/best_model.pth}"
OUTPUT="${2:-data/output/exported_models/model_docker.tflite}"
PRECISION="${3:-fp32}"  # 精度参数：fp32, fp16, int8

# Docker 镜像名称
DOCKER_IMAGE="image-classifier-tflite:latest"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🐳 Docker TFLite 模型导出${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "项目根目录: ${PROJECT_ROOT}"
echo -e "Checkpoint:  ${CHECKPOINT}"
echo -e "输出路径:    ${OUTPUT}"
echo -e "精度:        ${PRECISION}"
echo ""

# ============================================
# 步骤 1: 检查 Docker 是否可用
# ============================================
echo -e "${YELLOW}[1/4]${NC} 检查 Docker..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安装${NC}"
    echo ""
    echo "请安装 Docker Desktop:"
    echo "  - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "  - Windows: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
fi

if ! docker version &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker 未运行${NC}"
    echo ""
    echo "请启动 Docker Desktop 后重试"
    exit 1
fi

echo -e "${GREEN}✓ Docker 可用${NC}"
echo ""

# ============================================
# 步骤 2: 验证 Checkpoint 文件
# ============================================
echo -e "${YELLOW}[2/4]${NC} 验证 Checkpoint..."

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ Checkpoint 文件不存在: ${CHECKPOINT}${NC}"
    echo ""
    echo "可用的 checkpoint:"
    ls -lh data/output/checkpoints/*.pth 2>/dev/null || echo "  (无可用 checkpoint)"
    exit 1
fi

CHECKPOINT_SIZE=$(ls -lh "$CHECKPOINT" | awk '{print $5}')
echo -e "${GREEN}✓ Checkpoint 存在 (${CHECKPOINT_SIZE})${NC}"
echo ""

# ============================================
# 步骤 3: 构建 Docker 镜像（如果需要）
# ============================================
echo -e "${YELLOW}[3/4]${NC} 准备 Docker 镜像..."

if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo -e "  ${BLUE}首次使用，正在构建 Docker 镜像...${NC}"
    echo -e "  ${BLUE}(这可能需要 5-10 分钟)${NC}"
    echo ""

    # 检测当前系统架构并设置正确的平台
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        PLATFORM="linux/arm64"
        echo -e "  ${BLUE}检测到 ARM 架构，使用 $PLATFORM${NC}"
    else
        PLATFORM="linux/amd64"
        echo -e "  ${BLUE}检测到 x86 架构，使用 $PLATFORM${NC}"
    fi
    echo ""

    # 使用 --platform 明确指定目标平台
    docker build --platform "$PLATFORM" -t "$DOCKER_IMAGE" -f docker/Dockerfile .

    echo ""
    echo -e "${GREEN}✓ 镜像构建完成${NC}"
else
    echo -e "${GREEN}✓ 镜像已存在${NC}"
fi
echo ""

# ============================================
# 步骤 4: 运行导出
# ============================================
echo -e "${YELLOW}[4/4]${NC} 开始导出..."
echo ""

# 准备路径
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR=$(dirname "$OUTPUT")
OUTPUT_NAME=$(basename "$OUTPUT")

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取绝对路径
CHECKPOINT_DIR_ABS=$(cd "$CHECKPOINT_DIR" && pwd)
OUTPUT_DIR_ABS=$(cd "$OUTPUT_DIR" && pwd)

echo -e "  ${BLUE}运行 Docker 容器...${NC}"
echo ""

# 检测平台（用于 docker run）
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    PLATFORM="linux/arm64"
else
    PLATFORM="linux/amd64"
fi

# 运行 Docker 容器
docker run --rm \
    --platform "$PLATFORM" \
    -v "$PROJECT_ROOT/src:/workspace/src:ro" \
    -v "$CHECKPOINT_DIR_ABS:/workspace/checkpoints:ro" \
    -v "$OUTPUT_DIR_ABS:/workspace/output:rw" \
    "$DOCKER_IMAGE" \
    python -c "
import sys
sys.path.insert(0, '/workspace')

from src.models.model_factory import load_model_from_checkpoint
from src.export.tflite_exporter import TFLiteExporter

print('加载模型 checkpoint...')
model, ckpt = load_model_from_checkpoint('/workspace/checkpoints/$CHECKPOINT_NAME')

# 获取配置
class_names = None
if 'config' in ckpt and hasattr(ckpt['config'], 'class_names'):
    class_names = ckpt['config'].class_names

print('初始化导出器...')
exporter = TFLiteExporter(model, img_size=224, class_names=class_names)

print('开始导出...')
exporter.export('/workspace/output/$OUTPUT_NAME', precision='$PRECISION')

print('\n✅ 导出完成！')
"

# ============================================
# 完成
# ============================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ TFLite 导出成功！${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "输出文件: ${GREEN}${OUTPUT}${NC}"

if [ -f "$OUTPUT" ]; then
    OUTPUT_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
    echo -e "文件大小: ${OUTPUT_SIZE}"
fi

echo ""
echo -e "💡 下一步:"
echo -e "  - 验证模型: uv run python -c \"import tensorflow as tf; interpreter = tf.lite.Interpreter('$OUTPUT'); print('✓ 模型加载成功')\""
echo -e "  - 部署到 Android 应用"
echo ""
