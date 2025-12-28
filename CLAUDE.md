# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ImageClassifierModel is a mobile image classification project that trains a deep learning model to classify smartphone screenshots into three categories: **Failure**, **Loading**, and **Success**. The model is designed for deployment on iOS and Android devices.

## Key Requirements

- Model must be lightweight for mobile deployment
- Must be easily callable from Android/iOS applications
- Use PyTorch for training (industry best practice)
- Leverage pre-trained models with fine-tuning when possible
- Project uses `uv` for environment and dependency management
- **Language convention**: Think in English, respond in Chinese (思考用英文，回复用中文)

## Dataset

Training data is located in `data/input/` directory. The system will recursively scan all subdirectories to find images organized by category folders:
- Images should be organized in folders where the folder name represents the class label
- Example structure: `data/input/DatasetName/Failure/`, `data/input/DatasetName/Loading/`, etc.
- All images with the same parent folder name will be grouped into the same class
- Currently contains 256 labeled mobile screenshots across 3 categories: Failure, Loading, Success

## Project Structure

```
ImageClassifierModel/
├── data/
│   ├── input/          # Training data (images organized by category)
│   └── output/         # Model outputs, evaluation reports, metrics
├── src/                # (To be created) Source code for training pipeline
├── notebooks/          # (Optional) Jupyter notebooks for exploration
└── models/             # (To be created) Saved model checkpoints
```

## Development Commands

### Environment Setup
```bash
# Initialize/sync dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### Training Pipeline
```bash
# Train the model (once implemented)
uv run python src/train.py

# Run with custom hyperparameters
uv run python src/train.py --epochs 50 --batch-size 16

# Evaluate model on validation set
uv run python src/evaluate.py
```

### Model Export
```bash
# Export to ONNX format (cross-platform)
uv run python src/export.py --format onnx

# Export to TensorFlow Lite (Android)
uv run python src/export.py --format tflite

# Export to CoreML (iOS)
uv run python src/export.py --format coreml
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_data_loading.py

# Run with coverage
uv run pytest --cov=src tests/
```

## Architecture Guidelines

### Model Selection Priorities
1. **MobileNetV2** or **EfficientNet-Lite** - Optimized for mobile inference
2. Use transfer learning from ImageNet pre-trained weights
3. Fine-tune only the final classification layers initially
4. Target model size: < 10 MB for mobile deployment

### Code Organization
- `src/data/` - Data loading, preprocessing, augmentation
- `src/models/` - Model architecture definitions
- `src/training/` - Training loop, validation, checkpointing
- `src/export/` - Model conversion utilities (ONNX, TFLite, CoreML)
- `src/utils/` - Helper functions, metrics, visualization
- `tests/` - Unit and integration tests

### Data Pipeline
- Train/validation/test split: 70/15/15
- Apply data augmentation (rotation, flip, brightness) to handle small dataset
- Normalize images to model input requirements (typically 224x224 or 320x320)
- Use PyTorch DataLoader with appropriate batch sizes

### Mobile Deployment Considerations
- Quantize models to INT8 for faster inference
- Test inference latency on target devices
- Include pre/post-processing in exported model when possible
- Document input format and output interpretation for mobile engineers

## Output Expectations

Models and artifacts should be saved to `data/output/`:
- Trained model checkpoints (.pth)
- Exported mobile models (.onnx, .tflite, .mlmodel)
- Training metrics and loss curves
- Confusion matrix and classification report
- Model performance summary (accuracy, precision, recall, F1-score per class)

