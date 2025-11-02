# Veriface Backend API

FastAPI backend for face recognition attendance system.

## Features

- **Face Registration**: Register users with face embeddings
- **Face Verification**: Verify faces against registered database
- **Liveness Detection**: Anti-spoof detection to prevent photo/video attacks
- **Emotion Recognition**: Detect 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
- **Multiple Modes**: Support mock, heuristic, and ONNX inference modes
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

Place your ONNX models in `app/models/`:
- `embedding_A.onnx` - Face embedding model A (512-D output)
- `embedding_B.onnx` - Face embedding model B (512-D output)
- `emotion.onnx` - Emotion classification (7 classes)
- `liveness.onnx` - Liveness detection (optional, uses heuristic if missing)

**Note**: Models are already downloaded from ONNX Model Zoo. If you see IR version errors, convert them:
```bash
python scripts/convert_onnx_version.py
```

After training on Kaggle, export models as ONNX and replace them. If models are IR version 12, convert them first.

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
- Set MODE=onnx
- Tạo các thư mục cần thiết
- Start server tại http://localhost:8000

**Manual (nếu cần):**
```bash
set MODE=onnx
uvicorn app.main:app --reload --port 8000
```

**Available modes:**
- `mock` - Fake outputs for testing
- `heur` - OpenCV-based heuristics  
- `onnx` - Real ONNX model inference (default trong run.bat)

### 4. Test API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Configuration Modes

- **`mock`** (default): Deterministic fake outputs for testing
- **`heur`**: OpenCV-based heuristics (Haar cascade, variance-based liveness)
- **`onnx`**: Real ONNX model inference (requires model files in `app/models/`)

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
