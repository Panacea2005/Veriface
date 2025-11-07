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
  check_type?: "check-in" | "check-out"
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

export interface AttendanceRecord {
  timestamp: string
  user_id: string
  type?: "check-in" | "check-out"
  match_score: number
  liveness_score: number
  emotion_label?: string
  emotion_confidence?: number
  user_name?: string
  department?: string
}

export interface AttendanceResponse {
  records: AttendanceRecord[]
  count: number
}

export interface AttendanceStats {
  total_records: number
  unique_users: number
  by_date: Record<string, number>
  by_user: Record<string, number>
  by_emotion: Record<string, number>
  by_type: Record<string, number>
  daily_trends: Record<string, number>
}

export interface UserMetadata {
  name?: string
  department?: string
  email?: string
}

/**
 * Get attendance records
 */
export async function getAttendance(params?: {
  limit?: number
  userId?: string
  dateFrom?: string
  dateTo?: string
}): Promise<AttendanceResponse> {
  const searchParams = new URLSearchParams()
  if (params?.limit) searchParams.append("limit", params.limit.toString())
  if (params?.userId) searchParams.append("user_id", params.userId)
  if (params?.dateFrom) searchParams.append("date_from", params.dateFrom)
  if (params?.dateTo) searchParams.append("date_to", params.dateTo)

  const url = `${API_BASE_URL}/api/attendance${searchParams.toString() ? `?${searchParams.toString()}` : ""}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error("Failed to fetch attendance records")
  }
  return response.json()
}

/**
 * Get attendance statistics
 */
export async function getAttendanceStats(params?: {
  userId?: string
  dateFrom?: string
  dateTo?: string
}): Promise<AttendanceStats> {
  const searchParams = new URLSearchParams()
  if (params?.userId) searchParams.append("user_id", params.userId)
  if (params?.dateFrom) searchParams.append("date_from", params.dateFrom)
  if (params?.dateTo) searchParams.append("date_to", params.dateTo)

  const url = `${API_BASE_URL}/api/attendance/stats${searchParams.toString() ? `?${searchParams.toString()}` : ""}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error("Failed to fetch attendance statistics")
  }
  return response.json()
}

/**
 * Export attendance records
 */
export async function exportAttendance(params?: {
  format?: "csv" | "json"
  userId?: string
  dateFrom?: string
  dateTo?: string
}): Promise<Blob> {
  const searchParams = new URLSearchParams()
  searchParams.append("format", params?.format || "csv")
  if (params?.userId) searchParams.append("user_id", params.userId)
  if (params?.dateFrom) searchParams.append("date_from", params.dateFrom)
  if (params?.dateTo) searchParams.append("date_to", params.dateTo)

  const url = `${API_BASE_URL}/api/attendance/export?${searchParams.toString()}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error("Failed to export attendance")
  }
  return response.blob()
}

/**
 * Get user metadata
 */
export async function getUserMetadata(): Promise<Record<string, UserMetadata>> {
  const response = await fetch(`${API_BASE_URL}/api/users/metadata`)
  if (!response.ok) {
    throw new Error("Failed to fetch user metadata")
  }
  return response.json()
}

/**
 * Set user metadata
 */
export async function setUserMetadata(
  userId: string,
  metadata: Partial<UserMetadata>
): Promise<{ status: string; user_id: string; metadata: UserMetadata }> {
  const formData = new FormData()
  formData.append("user_id", userId)
  if (metadata.name) formData.append("name", metadata.name)
  if (metadata.department) formData.append("department", metadata.department)
  if (metadata.email) formData.append("email", metadata.email)

  const response = await fetch(`${API_BASE_URL}/api/users/metadata`, {
    method: "POST",
    body: formData,
  })
  if (!response.ok) {
    throw new Error("Failed to set user metadata")
  }
  return response.json()
}

/**
 * Delete user metadata
 */
export async function deleteUserMetadata(userId: string): Promise<{ status: string; deleted: string }> {
  const response = await fetch(`${API_BASE_URL}/api/users/metadata/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  })
  if (!response.ok) {
    throw new Error("Failed to delete user metadata")
  }
  return response.json()
}
