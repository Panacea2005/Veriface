# Models Directory

Place your ONNX models here.

## Required Models

1. **embedding_A.onnx** - Face embedding model A (output: 512-D vector)
2. **embedding_B.onnx** - Face embedding model B (output: 512-D vector)
3. **emotion.onnx** - Emotion classification (output: 7 classes logits)
4. **expclip_emotion.onnx** (optional, preferred) - ExpCLIP L14 emotion classifier
5. **liveness.onnx** - Liveness/anti-spoof detection (output: single score [0-1])

## Current Status

- ✅ `embedding_A.onnx` - Downloaded from ONNX Model Zoo (ArcFace ResNet-100)
- ✅ `embedding_B.onnx` - Same as A (replace with your trained model)
- ⚠️ `emotion.onnx` - Download manually (see scripts/DOWNLOAD_EMOTION_MODEL.md)
- ✅ If you have `ExpCLIP_L14_model.pth`, convert to ONNX (224x224 RGB, CLIP normalization) and save as `expclip_emotion.onnx` (backend will auto-prefer this)
- ⚠️ `liveness.onnx` - Not available publicly (backend uses heuristic mode)

## After Training on Kaggle

1. Export your trained models as ONNX format
2. Place them in this directory with the names above
3. Restart the backend: `export MODE=onnx && uvicorn app.main:app --reload`

## Model Requirements

- **Input format**: 
  - Embedding/Liveness: RGB image, 112x112, normalized [0-1], CHW format with batch dimension
  - Emotion (generic): Grayscale image, 64x64, normalized [0-1], CHW format
  - Emotion (ExpCLIP): RGB image, 224x224, CLIP mean/std normalization, CHW format
- **Output format**:
  - Embedding: 512-D float32 vector (will be normalized)
  - Emotion: 7-D float32 logits (will be softmaxed)
  - Liveness: Single float32 score [0-1]
