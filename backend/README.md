# Veriface Backend API

FastAPI backend for face recognition attendance system.

## Features

- **Face Registration**: Register users with face embeddings
- **Face Verification**: Verify faces against registered database
- **Liveness Detection**: Anti-spoof detection to prevent photo/video attacks
- **Emotion Recognition**: Detect 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
- **Embedding Models**: PyTorch ArcFace (.pth) with DeepFace fallback
- **Two Embedding Models**: Switch between Model A and B for comparison
- **Similarity Metrics**: Cosine similarity and Euclidean distance
- **RESTful API**: Full REST API with automatic documentation

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Models

Place your trained PyTorch model in `app/models/`:
- `ms1mv3_arcface_r100_fp16.pth` - Face embedding model (512-D output)

The system will automatically use the PyTorch model if available, otherwise falls back to DeepFace ArcFace.
Emotion and Liveness use DeepFace (no additional models required).

### 3. Run Backend

**Windows (PowerShell):**
```powershell
cd backend
.\run.bat
```

**Windows (CMD):**
```cmd
cd backend
run.bat
```

Script sẽ tự động:
- Kiểm tra Python và dependencies
- Cài đặt dependencies nếu thiếu
- Tạo các thư mục cần thiết
- Start server tại http://localhost:8000

**Manual (nếu cần):**
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Model Architecture

- **Face Embedding**: PyTorch ArcFace ResNet-100 (512-D vectors)
  - Automatically loads from `app/models/ms1mv3_arcface_r100_fp16.pth`
  - Falls back to DeepFace ArcFace if PyTorch model unavailable
- **Face Detection**: DeepFace with OpenCV backend
- **Liveness Detection**: DeepFace anti-spoofing
- **Emotion Recognition**: DeepFace emotion classifier

## API Endpoints

### `GET /health`
Health check - returns `{status: "ok"}`

### `POST /api/register`
Register a new user face.

**Parameters:**
- `user_id` (string, required): User identifier
- `model` (string, optional): "A" or "B" (default: "A")
- `image` (file, optional): Multipart image file
- `image_b64` (string, optional): Base64 encoded image

**Returns:**
```json
{
  "status": "success",
  "user_id": "user123",
  "model": "A",
  "embedding_shape": [512]
}
```

### `POST /api/verify`
Verify a face against registered database.

**Parameters:**
- `model` (string, optional): "A" or "B" (default: "A")
- `metric` (string, optional): "cosine" or "euclidean" (default: "cosine")
- `image` (file, optional): Multipart image file
- `image_b64` (string, optional): Base64 encoded image

**Returns:**
```json
{
  "liveness": {
    "score": 0.85,
    "passed": true
  },
  "matched_id": "user123",
  "score": 0.87,
  "metric": "cosine",
  "threshold": 0.75,
  "emotion_label": "happy",
  "emotion_confidence": 0.92
}
```

## Folder Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── core/             # Configuration files
│   │   ├── config.py     # Settings (paths, CORS)
│   │   ├── thresholds.yaml
│   │   └── label_map.json
│   ├── routers/          # API endpoints
│   │   ├── health.py
│   │   ├── register.py
│   │   └── verify.py
│   ├── pipelines/        # Model pipelines
│   │   ├── detector.py   # Face detection & alignment
│   │   ├── liveness.py   # Liveness detection
│   │   ├── embedding.py  # Face embedding extraction
│   │   ├── emotion.py    # Emotion classification
│   │   ├── similarity.py # Similarity matching
│   │   └── registry.py   # User registry (JSON store)
│   ├── models/           # PyTorch model files (place here)
│   └── store/            # Registry JSON storage
└── requirements.txt
```

## After Training on Kaggle

1. Export your trained PyTorch model as `.pth` format
2. Place it in `app/models/` as `ms1mv3_arcface_r100_fp16.pth`
3. Restart backend - model will be automatically loaded

The model should be an ArcFace ResNet-100 architecture with 512-D output embeddings.
