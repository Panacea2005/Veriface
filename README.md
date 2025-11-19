<div align="center">
  <img src="/public/logo.png" alt="Veriface Logo" width="120" height="120">
  
  # Veriface
  
  **Face Recognition Attendance System**
  
  [![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
  [![Next.js](https://img.shields.io/badge/Next.js-16.0-black.svg)](https://nextjs.org/)
  [![React](https://img.shields.io/badge/React-19.2-61DAFB.svg)](https://reactjs.org/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## Overview

Veriface is a comprehensive face recognition attendance system that combines advanced deep learning models with a modern web interface. The system provides real-time face verification, attendance tracking, liveness detection, and emotion analysis capabilities.

## Features

- **Face Recognition**: High-accuracy face verification using custom-trained PyTorch models
- **Attendance Management**: Automated check-in/check-out tracking with timestamp logging
- **Liveness Detection**: Anti-spoofing protection to prevent photo/video attacks
- **Emotion Analysis**: Real-time emotion detection and analytics
- **Web Interface**: Modern, responsive UI built with Next.js and React
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Real-time Processing**: Webcam-based face capture and verification
- **Registry Management**: User registration and embedding storage system

## Tech Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.17
- **Face Recognition**: Custom ArcFace models, DeepFace
- **Image Processing**: OpenCV, Pillow
- **Data Storage**: JSON-based registry and logs

### Frontend
- **Framework**: Next.js 16.0
- **UI Library**: React 19.2
- **Language**: TypeScript 5.0
- **Styling**: Tailwind CSS 4.1
- **UI Components**: Radix UI, shadcn/ui
- **Animations**: Framer Motion

## Project Structure

```
Veriface/
├── backend/
│   ├── app/
│   │   ├── core/           # Configuration and settings
│   │   ├── models/         # Trained model checkpoints
│   │   ├── pipelines/      # Face detection, embedding, similarity
│   │   ├── routers/        # API endpoints
│   │   └── store/          # Registry and logs (gitignored)
│   ├── requirements.txt
│   └── run.bat            # Windows startup script
├── components/            # React components
├── lib/                   # Utilities and API client
├── public/                # Static assets
└── app/                   # Next.js app directory
```

## Prerequisites

- Python 3.12 or higher
- Node.js 18+ and npm/pnpm
- Windows/Linux/macOS
- Webcam (for real-time verification)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd Veriface/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment variables (optional):
Create a `.env` file in the `backend` directory:
```env
MODEL_TYPE=A
SIMILARITY_METRIC=cosine
ENABLE_DEEPFACE=1
MODEL_TORCH_WEIGHT=0.15
MODEL_DEEPFACE_WEIGHT=0.85
CORS_ORIGINS=http://localhost:3000
```

6. Place model checkpoints in `backend/app/models/`:
   - `modelA_best.pth` (required)
   - `modelB_best.pth` (optional, if using Model B)

### Frontend Setup

1. Navigate to the project root:
```bash
cd Veriface
```

2. Install dependencies:
```bash
npm install
# or
pnpm install
```

## Running the Application

### Backend

**Windows:**
```bash
cd Veriface/backend
run.bat
```

**Manual start:**
```bash
cd Veriface/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend

```bash
cd Veriface
npm run dev
# or
pnpm dev
```

The web interface will be available at `http://localhost:3000`

## API Documentation

Once the backend is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /api/register` - Register a new user with face embedding
- `POST /api/verify` - Verify a face against the registry
- `GET /api/registry` - Get all registered users
- `GET /api/attendance` - Get attendance records
- `GET /api/emotion/analytics` - Get emotion analytics
- `GET /api/health` - Health check endpoint

## Configuration

### Model Selection

Set `MODEL_TYPE` in `.env`:
- `A`: Use Model A (default)
- `B`: Use Model B

### Similarity Metric

Set `SIMILARITY_METRIC` in `.env`:
- `cosine`: Cosine similarity (default)
- `euclidean`: Euclidean distance

### Thresholds

Edit `backend/app/core/thresholds.yaml` to adjust similarity thresholds.

## Usage

1. **Register Users**: Use the registration interface to capture and register user faces
2. **Verify Faces**: Use the verification interface to check faces against the registry
3. **View Attendance**: Access attendance history and analytics
4. **Monitor Emotions**: View real-time emotion detection results

## Development

### Backend Development

The backend follows a modular pipeline architecture:
- `detector.py`: Face detection and alignment
- `embedding.py`: Feature extraction using PyTorch models
- `similarity.py`: Similarity computation and matching
- `liveness.py`: Anti-spoofing detection
- `emotion.py`: Emotion recognition
- `registry.py`: User data management

### Frontend Development

The frontend uses:
- Next.js App Router for routing
- React Server Components where applicable
- TypeScript for type safety
- Tailwind CSS for styling
- shadcn/ui components for UI elements

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI for the excellent web framework
- Next.js team for the React framework
- All open-source contributors whose libraries made this project possible

---

<div align="center">
  <p>Built with precision and attention to detail</p>
</div>

