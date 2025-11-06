# Veriface Backend API

FastAPI backend for face recognition attendance system.

## Features

- **Face Registration**: Register users with face embeddings
- **Face Verification**: Verify faces against registered database
- **Liveness Detection**: Anti-spoof detection to prevent photo/video attacks
- **Emotion Recognition**: Detect 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
- **Modes**: Heuristic and ONNX inference (no mock)
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

Place your ONNX models in `app/models/` if you use ONNX mode:
- `embedding_A.onnx` - Face embedding model A (512-D output)
- `embedding_B.onnx` - Face embedding model B (512-D output)
Emotion and Liveness use DeepFace (no ONNX required)

After training, export embeddings as ONNX if needed for `MODE=onnx`. Emotion/Liveness are DeepFace-based and require no ONNX.

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
- Set MODE=heur (DeepFace for detection/liveness/emotion)
- Tạo các thư mục cần thiết
- Start server tại http://localhost:8000

**Manual (nếu cần):**
```bash
set MODE=heur
uvicorn app.main:app --reload --port 8000
```

**Available modes:**
- `heur` - DeepFace-based detection/liveness/emotion; heuristic fallbacks
- `onnx` - Embedding via ONNXRuntime; detection/liveness/emotion still DeepFace

### 4. Test API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Configuration Modes

- **`heur`** (default): DeepFace for detection/liveness/emotion
- **`onnx`**: Embeddings via ONNX (requires `embedding_*.onnx`), DeepFace for others

## API Endpoints

### `GET /health`
Health check - returns `{status: "ok", mode: MODE}`

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

### `GET /api/roc`
Get ROC curve metrics (AUC, EER, accuracy, precision, recall).

## Folder Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── core/             # Configuration files
│   │   ├── config.py     # Settings (MODE, paths, CORS)
│   │   ├── thresholds.yaml
│   │   ├── preprocess.json
│   │   └── label_map.json
│   ├── routers/          # API endpoints
│   │   ├── health.py
│   │   ├── register.py
│   │   ├── verify.py
│   │   └── metrics.py
│   ├── pipelines/        # Model pipelines
│   │   ├── detector.py   # Face detection & alignment
│   │   ├── liveness.py   # Liveness detection
│   │   ├── embedding.py  # Face embedding extraction
│   │   ├── emotion.py    # Emotion classification
│   │   ├── similarity.py # Similarity matching
│   │   └── registry.py   # User registry (JSON store)
│   ├── models/           # ONNX model files (place here)
│   └── store/            # Registry JSON storage
└── requirements.txt
```

## After Training on Kaggle

1. Export your trained models as ONNX format
2. Place them in `app/models/` with correct names
3. Restart backend - models will be automatically loaded

See `app/models/README.md` for detailed model requirements.
