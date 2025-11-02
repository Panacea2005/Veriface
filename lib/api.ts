const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface VerifyResponse {
  liveness: {
    score: number
    passed: boolean
  }
  matched_id: string | null
  score: number | null
  metric: string
  threshold: number | null
  emotion_label: string
  emotion_confidence: number
  all_scores?: Array<{
    user_id: string
    score: number
    percentage: number
    embeddings_count: number
  }>
}

export interface RegisterResponse {
  status: string
  user_id: string
  model: string
  embedding_shape: number[]
}

export interface HealthResponse {
  status: string
  mode: string
  registry?: {
    accessible: boolean
    path?: string
    users_count?: number
    total_embeddings?: number
    users?: string[]
    error?: string
  }
}

export interface MetricsResponse {
  auc: number
  eer: number
  accuracy: number
  precision: number
  recall: number
  // Optional detailed metrics
  faceVerification?: {
    tp: number
    fp: number
    fn: number
    tn: number
    tpr: number
    fpr: number
  }
  liveness?: {
    apcer: number
    bpcer: number
    acer: number
  }
  emotion?: {
    macroF1: number
    accuracy: number
  }
}

/**
 * Convert File/Blob to base64 string
 */
export async function fileToBase64(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result)
      } else {
        reject(new Error("Failed to convert file to base64"))
      }
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * Health check
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error("Health check failed")
  }
  return response.json()
}

/**
 * Register a new face
 */
export async function registerFace(
  userId: string,
  image: File | Blob,
  model: "A" | "B" = "A"
): Promise<RegisterResponse> {
  const formData = new FormData()
  formData.append("user_id", userId)
  formData.append("model", model)
  formData.append("image", image)

  const response = await fetch(`${API_BASE_URL}/api/register`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Registration failed" }))
    throw new Error(error.detail || "Registration failed")
  }

  return response.json()
}

/**
 * Verify a face
 */
export async function verifyFace(
  image: File | Blob,
  model: "A" | "B" = "A",
  metric: "cosine" | "euclidean" = "cosine"
): Promise<VerifyResponse> {
  const formData = new FormData()
  formData.append("model", model)
  formData.append("metric", metric)
  formData.append("image", image)

  const response = await fetch(`${API_BASE_URL}/api/verify`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Verification failed" }))
    throw new Error(error.detail || "Verification failed")
  }

  return response.json()
}

/**
 * Get ROC metrics
 */
export async function getMetrics(): Promise<MetricsResponse | null> {
  const response = await fetch(`${API_BASE_URL}/api/roc`)
  if (!response.ok) {
    if (response.status === 404) {
      // No metrics available yet
      return null
    }
    throw new Error("Failed to fetch metrics")
  }
  return response.json()
}

