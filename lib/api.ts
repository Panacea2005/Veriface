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
  emotion_probs?: Record<string, number>
  bbox?: { x: number; y: number; w: number; h: number }
  all_scores?: Array<{
    user_id: string
    score: number
    percentage: number
    embeddings_count: number
  }>
}

export interface EmotionResponse {
  label: string
  confidence: number
  probs: Record<string, number>
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

export interface RegistryResponse {
  users: string[]
  counts: Record<string, number>
  embeddings2d: Array<{ user_id: string; index: number; x: number; y: number }>
  vectors?: Record<string, number[][]>
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

export async function analyzeEmotion(image: File | Blob): Promise<EmotionResponse> {
  const formData = new FormData()
  formData.append("image", image)
  const response = await fetch(`${API_BASE_URL}/api/emotion`, { method: "POST", body: formData })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Emotion analysis failed" }))
    throw new Error(error.detail || "Emotion analysis failed")
  }
  return response.json()
}

/**
 * Fetch registry identities and (optionally) embeddings/2D projection
 */
export async function fetchRegistry(params?: {
  includeVectors?: boolean
  project?: "none" | "pca2d"
  limitPerUser?: number
}): Promise<RegistryResponse> {
  const includeVectors = params?.includeVectors ? "true" : "false"
  const project = params?.project ?? "pca2d"
  const limit = params?.limitPerUser ?? 200
  const url = `${API_BASE_URL}/api/registry?include_vectors=${includeVectors}&project=${project}&limit_per_user=${limit}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error("Failed to fetch registry")
  }
  return response.json()
}

export async function deleteRegistryUser(userId: string): Promise<{ status: string; deleted: string }> {
  const response = await fetch(`${API_BASE_URL}/api/registry/${encodeURIComponent(userId)}`, { method: "DELETE" })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Delete user failed" }))
    throw new Error(err.detail || "Delete user failed")
  }
  return response.json()
}

export async function deleteRegistryEmbedding(userId: string, index: number): Promise<{ status: string; user_id: string; deleted_index: number }> {
  const response = await fetch(`${API_BASE_URL}/api/registry/${encodeURIComponent(userId)}/${index}`, { method: "DELETE" })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Delete embedding failed" }))
    throw new Error(err.detail || "Delete embedding failed")
  }
  return response.json()
}

export async function clearRegistry(): Promise<{ status: string; message: string; deleted_users: number }> {
  const response = await fetch(`${API_BASE_URL}/api/registry`, { method: "DELETE" })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Clear registry failed" }))
    throw new Error(err.detail || "Clear registry failed")
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

