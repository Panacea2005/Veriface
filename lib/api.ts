const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface VerifyResponse {
  liveness: {
    score: number
    passed: boolean
  }
  matched_id: string | null
  matched_name?: string | null
  score: number | null
  percentage?: number | null  // Percentage (0-100) for the best match score
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
  age?: number
  gender?: string
  gender_confidence?: number
  race?: string
  race_confidence?: number
}

export interface LivenessResponse {
  score: number
  passed: boolean
  is_real: boolean
  processing_time_ms: number
  status: "success" | "error"
  message: string
}

export interface ProcessingResult {
  image_index: number
  status: "processed" | "skipped" | "error"
  reason?: string
  error?: string
  has_torch?: boolean
  torch_norm?: number
  torch_mean?: number
  torch_std?: number
  similarity_to_first?: number
}

export interface RegisterResponse {
  status: string
  user_id: string
  name: string
  embedding_count: number
  model: string
  embedding_shape: number[]
  images_processed?: number
  images_total?: number
  embeddings_saved?: number
  processing_results?: ProcessingResult[]  // Processing results for all images (for visualization)
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
  names: Record<string, string>  // user_id -> name mapping
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
  name: string,
  image: File | Blob,
  userId?: string, // Optional: if provided, adds embedding to existing user
): Promise<RegisterResponse> {
  const formData = new FormData()
  formData.append("name", name)
  if (userId) {
    formData.append("user_id", userId)
  }
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
 * Register a new face with multiple images (batch registration - each image saved separately)
 */
export async function registerFaceBatch(
  name: string,
  images: File[] | Blob[],
  userId?: string, // Optional: if provided, adds embeddings to existing user
): Promise<RegisterResponse> {
  const formData = new FormData()
  formData.append("name", name)
  if (userId) {
    formData.append("user_id", userId)
  }
  
  images.forEach((image) => {
    formData.append("images", image)
  })

  const response = await fetch(`${API_BASE_URL}/api/register/batch`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Batch registration failed" }))
    throw new Error(error.detail || "Batch registration failed")
  }

  return response.json()
}

/**
 * Verify a face
 */
export async function verifyFace(
  image: File | Blob,
): Promise<VerifyResponse> {
  const formData = new FormData()
  formData.append("image", image)

  const response = await fetch(`${API_BASE_URL}/api/verify`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Verification failed" }))
    const errorMessage = error.detail || "Verification failed"
    // Create error object that won't trigger Next.js error boundary for expected errors
    const err = new Error(errorMessage)
    // Mark as handled to prevent Next.js error boundary
    ;(err as any).handled = true
    throw err
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
 * Real-time liveness check (optimized for streaming)
 */
export async function checkLivenessRealtime(image: File | Blob): Promise<LivenessResponse> {
  const formData = new FormData()
  formData.append("image", image)
  
  const response = await fetch(`${API_BASE_URL}/api/liveness/realtime`, { 
    method: "POST", 
    body: formData 
  })
  
  // Always return JSON (endpoint never throws HTTP errors)
  const result = await response.json()
  return result as LivenessResponse
}

/**
 * Check if a name already exists in the registry
 */
export async function checkNameExists(name: string): Promise<boolean> {
  try {
    const registry = await fetchRegistry({ includeVectors: false, project: "none", limitPerUser: 1 })
    const existingNames = Object.values(registry.names).map(n => n.toLowerCase().trim())
    return existingNames.includes(name.toLowerCase().trim())
  } catch (error) {
    console.error("[API] Failed to check name existence:", error)
    return false // If check fails, allow registration to proceed
  }
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
  name?: string  // User's name from registry (new field)
  type?: "check-in" | "check-out"
  match_score: number
  liveness_score: number
  emotion_label?: string
  emotion_confidence?: number
  user_name?: string  // Legacy field (kept for compatibility)
  department?: string
}

export interface AttendanceResponse {
  records: AttendanceRecord[]
  count: number
}

export interface AttendanceStats {
  total_records: number
  total_entries: number  // Alias for total_records
  unique_users: number
  check_ins: number
  check_outs: number
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
