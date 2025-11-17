"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Camera, RefreshCw, Upload, CheckCircle2, AlertCircle, RotateCcw, ShieldAlert, LogIn, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RegisterDrawer } from "@/components/register-drawer"
import { motion } from "framer-motion"
import { verifyFace, checkHealth, analyzeEmotion, checkLivenessRealtime } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import type { VerifyResponse, HealthResponse, LivenessResponse } from "@/lib/api"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import RegistryDialog from "@/components/registry-dialog"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"

interface WebcamSectionProps {
  onVerifyResult: (result: VerifyResponse | null) => void
}

type Mode = "upload" | "webcam"

export function WebcamSection({ onVerifyResult }: WebcamSectionProps) {
  const [mode, setMode] = useState<Mode>("upload")
  const [isAutoCapture, setIsAutoCapture] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [isFlipped, setIsFlipped] = useState(true)
  const [registryInfo, setRegistryInfo] = useState<HealthResponse["registry"] | null>(null)
  
  // Webcam refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  const { toast } = useToast()
  const [registryOpen, setRegistryOpen] = useState(false)
  const [liveEmotion, setLiveEmotion] = useState<{ label: string; probs: Record<string, number>; age?: number; gender?: string; race?: string } | null>(null)
  const emotionTickRef = useRef<boolean>(false)
  const [liveLiveness, setLiveLiveness] = useState<LivenessResponse | null>(null)
  const livenessTickRef = useRef<boolean>(false)
  const [lastVerify, setLastVerify] = useState<VerifyResponse | null>(null)
  const [spoofDialogOpen, setSpoofDialogOpen] = useState(false)
  const [spoofMessage, setSpoofMessage] = useState<string>("")
  const webcamAreaRef = useRef<HTMLDivElement>(null)
  const [webcamSquareSize, setWebcamSquareSize] = useState<number | null>(null)

  // Load registry info on mount and listen for registry updates
  useEffect(() => {
    const fetchRegistryInfo = async () => {
      try {
        const health = await checkHealth()
        if (health.registry) {
          setRegistryInfo(health.registry)
        }
      } catch (error) {
        console.error("Failed to fetch registry info:", error)
      }
    }
    
    fetchRegistryInfo()
    
    // Listen for registry update events (triggered after registration)
    const handleRegistryUpdate = () => {
      fetchRegistryInfo()
    }
    window.addEventListener('registry-updated', handleRegistryUpdate)
    
    // Refresh every 10 seconds
    const interval = setInterval(fetchRegistryInfo, 10000)
    
    return () => {
      window.removeEventListener('registry-updated', handleRegistryUpdate)
      clearInterval(interval)
    }
  }, [])

  // Keep webcam square sized based on container width (height matches width)
  useEffect(() => {
    if (mode !== "webcam") {
      setWebcamSquareSize(null)
      return
    }
    const updateSize = () => {
      const el = webcamAreaRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const size = Math.max(0, Math.min(rect.width, rect.height))
      setWebcamSquareSize(size === 0 ? null : size)
    }
    updateSize()
    window.addEventListener("resize", updateSize)
    return () => {
      window.removeEventListener("resize", updateSize)
    }
  }, [mode])

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [])

  const startWebcam = useCallback(async () => {
    try {
      if (streamRef.current) {
        // Already started
        if (videoRef.current && videoRef.current.srcObject !== streamRef.current) {
          videoRef.current.srcObject = streamRef.current
        }
        try { await videoRef.current?.play() } catch {}
        return
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        }
      })
      streamRef.current = stream
      const video = videoRef.current
      if (video) {
        if (video.srcObject !== stream) {
          video.srcObject = stream
        }
        const onLoaded = () => {
          video.play().catch(() => {})
          video.removeEventListener('loadedmetadata', onLoaded)
        }
        video.addEventListener('loadedmetadata', onLoaded)
        // Fallback attempt
        try { await video.play() } catch {}
      }
    } catch (error) {
      toast({
        title: "Camera access denied",
        description: "Please allow camera access to use webcam mode",
        variant: "destructive",
      })
      setMode("upload")
    }
  }, [toast])

  // Initialize webcam
  useEffect(() => {
    if (mode === "webcam") {
      startWebcam()
    } else {
      stopWebcam()
    }
    
    return () => {
      stopWebcam()
    }
  }, [mode, startWebcam, stopWebcam])

  const drawSquareFrame = (
    ctx: CanvasRenderingContext2D,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement
  ) => {
    const videoWidth = video.videoWidth || 0
    const videoHeight = video.videoHeight || 0
    if (videoWidth === 0 || videoHeight === 0) return false
    const squareSize = Math.min(videoWidth, videoHeight)
    const offsetX = (videoWidth - squareSize) / 2
    const offsetY = (videoHeight - squareSize) / 2
    canvas.width = squareSize
    canvas.height = squareSize

    ctx.save()
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(
      video,
      offsetX,
      offsetY,
      squareSize,
      squareSize,
      0,
      0,
      canvas.width,
      canvas.height
    )
    ctx.restore()
    return true
  }

  // Realtime emotion loop (webcam only)
  useEffect(() => {
    if (mode !== "webcam") return
    const interval = setInterval(async () => {
      if (emotionTickRef.current) return
      if (!videoRef.current || !canvasRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      const drawn = drawSquareFrame(ctx, video, canvas)
      if (!drawn) return
      emotionTickRef.current = true
      canvas.toBlob(async (blob) => {
        try {
          if (!blob) return
          const file = new File([blob], "frame.jpg", { type: "image/jpeg" })
          const emo = await analyzeEmotion(file)
          const payload = { 
            label: emo.label, 
            probs: emo.probs || {},
            age: emo.age,
            gender: emo.gender,
            race: emo.race
          }
          setLiveEmotion(payload)
          try {
            window.dispatchEvent(new CustomEvent('live-emotion', { detail: payload }))
          } catch {}
        } catch (e) {
          // ignore transient errors
        } finally {
          emotionTickRef.current = false
        }
      }, "image/jpeg", 0.9)
    }, 800) // ~1.25 fps to balance load; adjust as needed
    return () => clearInterval(interval)
  }, [mode])

  // Realtime liveness loop (webcam only) - runs in parallel with emotion
  useEffect(() => {
    if (mode !== "webcam") return
    const interval = setInterval(async () => {
      if (livenessTickRef.current) return
      if (!videoRef.current || !canvasRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return
      
      // Create temporary canvas for liveness check
      const tempCanvas = document.createElement('canvas')
      const ctx = tempCanvas.getContext("2d")
      if (!ctx) return
      const drawn = drawSquareFrame(ctx, video, tempCanvas)
      if (!drawn) return
      
      livenessTickRef.current = true
      tempCanvas.toBlob(async (blob) => {
        try {
          if (!blob) return
          const file = new File([blob], "frame.jpg", { type: "image/jpeg" })
          const liveness = await checkLivenessRealtime(file)
          setLiveLiveness(liveness)
          
          // Show spoof dialog if spoof detected with high confidence
          if (!liveness.passed && liveness.score > 0.7) {
            setSpoofMessage(`Spoof detected with ${(liveness.score * 100).toFixed(0)}% confidence. Please use a real face.`)
            setSpoofDialogOpen(true)
          }
          
          // Dispatch event for other components
          try {
            window.dispatchEvent(new CustomEvent('live-liveness', { detail: liveness }))
          } catch {}
        } catch (e) {
          // Silently handle errors (network issues, etc.)
          console.warn('Liveness check failed:', e)
        } finally {
          livenessTickRef.current = false
        }
      }, "image/jpeg", 0.85)
    }, 1000) // 1 FPS for liveness (less frequent than emotion to reduce load)
    return () => clearInterval(interval)
  }, [mode])

  const ensureSquareImage = useCallback(async (file: File): Promise<File> => {
      const dataUrl: string = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          if (typeof reader.result === "string") {
            resolve(reader.result)
          } else {
            reject(new Error("Failed to read file"))
          }
        }
        reader.onerror = reject
        reader.readAsDataURL(file)
      })

      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image()
        image.onload = () => resolve(image)
        image.onerror = reject
        image.src = dataUrl
      })

      const size = Math.min(img.width, img.height)
      if (size <= 0) {
        return file
      }
      const offsetX = (img.width - size) / 2
      const offsetY = (img.height - size) / 2

      const canvas = document.createElement("canvas")
      canvas.width = canvas.height = size
      const ctx = canvas.getContext("2d")
      if (!ctx) return file

      ctx.drawImage(img, offsetX, offsetY, size, size, 0, 0, size, size)

      const blob: Blob | null = await new Promise((resolve) => {
        canvas.toBlob((b) => resolve(b), "image/jpeg", 0.95)
      })
      if (!blob) {
        return file
      }
      return new File([blob], file.name || "upload.jpg", { type: "image/jpeg" })
  }, [])

  const captureFromWebcam = () => {
    if (!videoRef.current || !canvasRef.current) return
    
    const video = videoRef.current
    const canvas = canvasRef.current
    
    const ctx = canvas.getContext("2d")
    if (ctx) {
      const drawn = drawSquareFrame(ctx, video, canvas)
      if (!drawn) return
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], "capture.jpg", { type: "image/jpeg" })
          setSelectedImage(file)
          const url = URL.createObjectURL(blob)
          setPreviewUrl(url)
          setZoom(1)
          setPosition({ x: 0, y: 0 })
        }
      }, "image/jpeg", 0.95)
    }
  }

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      try {
        const squared = await ensureSquareImage(file)
        setSelectedImage(squared)
        const url = URL.createObjectURL(squared)
        setPreviewUrl(url)
      } catch (error) {
        console.warn("Failed to normalize upload, using original file", error)
        setSelectedImage(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
      } finally {
        setZoom(1)
        setPosition({ x: 0, y: 0 })
      }
    }
  }

  const handleVerify = async () => {
    if (!selectedImage) {
      if (mode === "webcam") {
        captureFromWebcam()
        return
      } else {
        fileInputRef.current?.click()
        return
      }
    }

    setIsCapturing(true)
    try {
      let imageToSend: File = selectedImage
      if (mode === "upload") {
        try {
          imageToSend = await ensureSquareImage(selectedImage)
        } catch (e) {
          console.warn("Failed to square image, using original", e)
        }
      }
      const result = await verifyFace(imageToSend)
      onVerifyResult(result)
      setLastVerify(result)
      if (result.emotion_label) {
        setLiveEmotion({ label: result.emotion_label, probs: result.emotion_probs || {} })
      }
      toast({
        title: "Verification complete",
        description: result.matched_id 
          ? `Matched: ${result.matched_name || result.matched_id} (ID: ${result.matched_id}) - ${(result.score! * 100).toFixed(1)}%`
          : "No match found",
      })
    } catch (error) {
      // Prevent error from propagating to Next.js error boundary
      const errorMessage = error instanceof Error ? error.message : "Unknown error"
      
      // Check if it's a spoof detection error
      if (errorMessage.toLowerCase().includes("spoof") || errorMessage.toLowerCase().includes("liveness") || errorMessage.toLowerCase().includes("anti-spoof")) {
        // Show spoof dialog
        setSpoofMessage(errorMessage)
        setSpoofDialogOpen(true)
        // Don't log to console to avoid Next.js error
      } else {
        // Show toast for other errors
        toast({
          title: "Verification failed",
          description: errorMessage,
          variant: "destructive",
        })
      }
    } finally {
      setIsCapturing(false)
    }
  }

  // Use refs to avoid dependency issues
  const onVerifyResultRef = useRef(onVerifyResult)
  const toastRef = useRef(toast)
  const setSpoofDialogOpenRef = useRef(setSpoofDialogOpen)
  const setSpoofMessageRef = useRef(setSpoofMessage)
  
  useEffect(() => {
    onVerifyResultRef.current = onVerifyResult
    toastRef.current = toast
    setSpoofDialogOpenRef.current = setSpoofDialogOpen
    setSpoofMessageRef.current = setSpoofMessage
  }, [onVerifyResult, toast, setSpoofDialogOpen, setSpoofMessage])

  // Auto capture logic
  useEffect(() => {
    if (!isAutoCapture || mode !== "webcam") return

    const interval = setInterval(async () => {
      // Skip if already processing or video not ready
      if (isCapturing || !videoRef.current || !canvasRef.current) return
      
      const video = videoRef.current
      const canvas = canvasRef.current
      
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      
      // Flip horizontal for mirror effect
      ctx.save()
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)
      ctx.drawImage(video, 0, 0)
      ctx.restore()
      
      canvas.toBlob(async (blob) => {
        if (!blob) return
        
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" })
        setIsCapturing(true)
        
        try {
          const result = await verifyFace(file)
          onVerifyResultRef.current(result)
          setLastVerify(result)
          if (result.emotion_label) {
            setLiveEmotion({ label: result.emotion_label, probs: result.emotion_probs || {} })
          }
          toastRef.current({
            title: "Auto-verification complete",
            description: result.matched_id 
              ? `Matched: ${result.matched_name || result.matched_id} (ID: ${result.matched_id})`
              : "No match found",
          })
        } catch (error) {
          // Prevent error from propagating to Next.js error boundary
          const errorMessage = error instanceof Error ? error.message : "Unknown error"
          
          // Check if it's a spoof detection error
          if (errorMessage.toLowerCase().includes("spoof") || errorMessage.toLowerCase().includes("liveness") || errorMessage.toLowerCase().includes("anti-spoof")) {
            // Show spoof dialog
            setSpoofMessageRef.current(errorMessage)
            setSpoofDialogOpenRef.current(true)
            // Don't log to console to avoid Next.js error
          } else {
            // Only log non-spoof errors, don't show toast in auto-capture to avoid spam
            console.error("Auto-verification failed:", error)
          }
        } finally {
          setIsCapturing(false)
        }
      }, "image/jpeg", 0.95)
    }, 3000) // Capture every 3 seconds

    return () => clearInterval(interval)
  }, [isAutoCapture, mode, isCapturing])

  // Zoom controls
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5))
  }

  const handleReset = () => {
    // Reset zoom and position
    setZoom(1)
    setPosition({ x: 0, y: 0 })
    
    // Clear selected image and preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setSelectedImage(null)
    setPreviewUrl(null)
    
    // Clear file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    
    // If in webcam mode, restart webcam to show live feed again
    if (mode === "webcam" && videoRef.current) {
      // Webcam will automatically show when selectedImage is null
      // No need to restart stream, just clear the captured image
    }
  }

  // Drag handlers for panning when zoomed
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const matchScoreDisplay = lastVerify?.score != null ? `${(lastVerify.score * 100).toFixed(1)}%` : "—"
  const livenessPassed = lastVerify?.liveness?.passed
  const thresholdDisplay = lastVerify?.threshold != null ? `${(lastVerify.threshold * 100).toFixed(0)}%` : "—"
  const metricDisplay = lastVerify?.metric ? lastVerify.metric.toUpperCase() : "—"

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <Card className="flex flex-col overflow-hidden border border-border shadow-sm">
        <CardHeader className="pb-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle className="text-base font-semibold">Face Capture & Verification</CardTitle>
              <CardDescription className="text-xs">Upload image or capture from webcam</CardDescription>
              {/* Model selection removed; always using Model A */}
            </div>
            <div className="flex items-center gap-3">
              {registryInfo && (
               <div className="flex flex-col items-end gap-1 text-xs">
                 <div className="flex items-center gap-2">
                   <div className={`h-2 w-2 rounded-full ${registryInfo.accessible ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                   <span className="text-foreground/80 font-medium">
                     {registryInfo.accessible ? "Registry Active" : "Registry Inaccessible"}
                   </span>
                 </div>
                 {registryInfo.accessible && (
                   <div className="flex flex-col items-end gap-2 text-foreground/70">
                     <span>
                       {registryInfo.users_count || 0} user{registryInfo.users_count !== 1 ? 's' : ''} • {registryInfo.total_embeddings || 0} embeddings
                     </span>
                     <div className="flex items-center gap-2">
                       <Button size="sm" variant="outline" onClick={() => setRegistryOpen(true)}>Explore</Button>
                       <RegisterDrawer />
                     </div>
                   </div>
                 )}
               </div>
             )}
             </div>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          <RegistryDialog open={registryOpen} onOpenChange={setRegistryOpen} />
          {/* Main Layout: Left webcam, Right controls + emotion */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-stretch">
            {/* Left: Capture area (spans 2) */}
            <div className="lg:col-span-2 space-y-4 flex flex-col h-full min-h-[600px]">
                <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)} className="h-full flex flex-col">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="upload" className="gap-2">
                    <Upload className="h-4 w-4" />
                    Upload Image
                  </TabsTrigger>
                  <TabsTrigger value="webcam" className="gap-2">
                    <Camera className="h-4 w-4" />
                    Webcam
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="upload" className="mt-4 flex-1">
                  <div className="flex items-center justify-center w-full h-full">
                    <div className="relative w-full max-w-full aspect-square rounded-xl bg-muted border border-border overflow-hidden flex items-center justify-center">
                      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
                      {previewUrl ? (
                        <div 
                          ref={containerRef}
                          className="relative w-full h-full overflow-hidden cursor-move"
                          onMouseDown={handleMouseDown}
                          onMouseMove={handleMouseMove}
                          onMouseUp={handleMouseUp}
                          onMouseLeave={handleMouseUp}
                        >
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="w-full h-full object-cover"
                            style={{
                              transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                              transformOrigin: "center center",
                              transition: isDragging ? "none" : "transform 0.1s ease-out"
                            }}
                            draggable={false}
                          />
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-3 text-muted-foreground text-center px-4">
                          <Upload className="h-12 w-12 opacity-40 stroke-[1.5]" />
                          <span className="text-sm font-medium">Select an image to upload</span>
                          <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()} className="mt-2">Choose Image</Button>
                        </div>
                      )}
                      {isCapturing && (
                        <motion.div className="absolute inset-0 bg-accent/10 flex items-center justify-center z-10" initial={{ opacity: 0 }} animate={{ opacity: [0, 0.5, 0] }} transition={{ duration: 0.6 }}>
                          <span className="text-sm font-medium">Processing...</span>
                        </motion.div>
                      )}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="webcam" className="mt-4 flex-1">
                  <div ref={webcamAreaRef} className="relative flex items-center justify-center w-full h-full">
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div
                        className="relative rounded-xl border border-border overflow-hidden"
                        style={
                          webcamSquareSize
                            ? { width: webcamSquareSize, height: webcamSquareSize }
                            : { width: "100%", aspectRatio: "1 / 1" }
                        }
                      >
                        <div className="absolute inset-0 bg-muted opacity-60" />
                        <div className="absolute inset-3 rounded-lg border-2 border-border/60 border-dashed" />
                      </div>
                    </div>
                    <div
                      className="relative flex items-center justify-center overflow-hidden rounded-xl bg-muted border border-border"
                      style={
                        webcamSquareSize
                          ? { width: webcamSquareSize, height: webcamSquareSize }
                          : { width: "100%", aspectRatio: "1 / 1" }
                      }
                    >
                    {/* Emotion overlay badge */}
                    {liveEmotion && liveEmotion.probs && Object.keys(liveEmotion.probs).length > 0 && (
                      (() => {
                        const [k, v] = Object.entries(liveEmotion.probs).sort((a,b)=>b[1]-a[1])[0]
                        const em = k === "happy" ? "😊" : k === "sad" ? "😢" : k === "angry" ? "😠" : k === "surprise" ? "😲" : k === "fear" ? "😨" : k === "disgust" ? "🤢" : "😐"
                        return (
                          <div className="absolute top-2 left-2 z-10 rounded-full bg-background/85 backdrop-blur px-3 py-1 border border-border text-xs font-semibold capitalize flex items-center gap-2">
                            <span className="text-xl leading-none">{em}</span>
                            <span>{k} {(v*100).toFixed(0)}%</span>
                          </div>
                        )
                      })()
                    )}
                    {/* Real-time Liveness indicator */}
                    {liveLiveness && (
                      <div className="absolute top-2 right-2 z-10 rounded-full bg-background/85 backdrop-blur px-3 py-1.5 border border-border text-xs font-semibold flex items-center gap-2">
                        <ShieldAlert className={`h-3.5 w-3.5 ${
                          liveLiveness.passed 
                            ? "text-green-600 dark:text-green-400" 
                            : "text-red-600 dark:text-red-400"
                        }`} />
                        <span className={liveLiveness.passed 
                          ? "text-green-600 dark:text-green-400" 
                          : "text-red-600 dark:text-red-400"
                        }>
                          {liveLiveness.passed 
                            ? `✓ Real ${(liveLiveness.score * 100).toFixed(0)}%` 
                            : `✗ Spoof ${(liveLiveness.score * 100).toFixed(0)}%`
                          }
                        </span>
                      </div>
                    )}
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`w-full h-full object-cover ${selectedImage ? "hidden" : ""}`}
                      style={{ transform: isFlipped ? "scaleX(-1)" : "none" }}
                    />
                    {previewUrl && selectedImage && (
                      <div 
                        ref={containerRef}
                        className="relative w-full h-full overflow-hidden cursor-move"
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                      >
                        <img
                          src={previewUrl}
                          alt="Captured"
                          className="w-full h-full object-contain"
                          style={{
                            transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                            transformOrigin: "center center",
                            transition: isDragging ? "none" : "transform 0.1s ease-out"
                          }}
                          draggable={false}
                        />
                      </div>
                    )}
                    {isCapturing && (
                      <motion.div className="absolute inset-0 bg-secondary/15 flex items-center justify-center z-10" initial={{ opacity: 0 }} animate={{ opacity: [0, 0.5, 0] }} transition={{ duration: 0.6 }}>
                        <span className="text-sm font-medium">Processing...</span>
                      </motion.div>
                    )}
                    <canvas ref={canvasRef} className="hidden" />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>

            {/* Right: Status + Emotion + Controls (bento grid) */}
            <div className="flex flex-col gap-4 h-full">
              {/* Compact Verification Status */}
              <div className="rounded-lg border border-border p-3 space-y-3 flex-shrink-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold">Status</span>
                  {lastVerify ? (
                    lastVerify.liveness.passed ? (
                      <span className="inline-flex items-center gap-1 text-green-600 text-xs"><CheckCircle2 className="h-3.5 w-3.5" /> Live</span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-red-600 text-xs"><AlertCircle className="h-3.5 w-3.5" /> Spoof</span>
                    )
                  ) : (
                    <span className="text-xs text-foreground/60">Waiting...</span>
                  )}
                </div>
                {lastVerify && (
                  <div className="space-y-3 text-xs">
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-foreground/60">Matched Identity</span>
                        {lastVerify.matched_id ? (
                          lastVerify.check_type ? (
                            <Badge
                              variant={lastVerify.check_type === "check-out" ? "secondary" : "default"}
                              className="h-5 px-2 text-[10px] font-semibold font-mono"
                            >
                              {lastVerify.check_type === "check-out" ? "OUT" : "IN"}
                            </Badge>
                          ) : (
                            <Badge
                              variant="outline"
                              className="h-5 px-2 text-[10px] font-semibold bg-primary/10 text-primary border-primary/30"
                            >
                              Matched
                            </Badge>
                          )
                        ) : (
                          <Badge
                            variant="outline"
                            className="h-5 px-2 text-[10px] font-semibold bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/40"
                          >
                            No Match
                          </Badge>
                        )}
                      </div>
                      <p className={`text-sm font-semibold ${lastVerify.matched_id ? "text-foreground" : "text-foreground/60"}`}>
                        {lastVerify.matched_id ?? "No match detected"}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px] text-foreground/70 md:grid-cols-4">
                      <div className="rounded-lg border border-border/60 bg-background/80 px-3 py-2">
                        <span className="text-[10px] uppercase tracking-wide text-foreground/60">Match Score</span>
                        <span className="block text-sm font-semibold text-foreground">{matchScoreDisplay}</span>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/80 px-3 py-2">
                        <span className="text-[10px] uppercase tracking-wide text-foreground/60">Liveness</span>
                        <span
                          className={`block text-sm font-semibold ${
                            livenessPassed === undefined
                              ? "text-foreground"
                              : livenessPassed
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-600 dark:text-red-400"
                          }`}
                        >
                          {livenessPassed === undefined
                            ? "—"
                            : lastVerify.liveness?.score != null
                              ? `${(lastVerify.liveness.score * 100).toFixed(1)}% • ${livenessPassed ? "Passed" : "Failed"}`
                              : livenessPassed
                                ? "Passed"
                                : "Failed"}
                        </span>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/80 px-3 py-2">
                        <span className="text-[10px] uppercase tracking-wide text-foreground/60">Threshold</span>
                        <span className="block text-sm font-semibold text-foreground">{thresholdDisplay}</span>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/80 px-3 py-2">
                        <span className="text-[10px] uppercase tracking-wide text-foreground/60">Metric</span>
                        <span className="block text-sm font-semibold text-foreground">{metricDisplay}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Realtime Emotion */}
              <div className="rounded-lg border border-border p-3 flex-1 flex flex-col">
                <div className="text-sm font-semibold mb-3">
                  <span>Emotion (real-time)</span>
                </div>
                {liveEmotion && liveEmotion.probs && Object.keys(liveEmotion.probs).length > 0 ? (
                  <div className="space-y-2 min-h-44 flex-1">
                    {Object.entries(liveEmotion.probs)
                      .sort((a, b) => b[1] - a[1])
                      .map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2">
                          <div className={`text-xs w-20 ${liveEmotion.label === k ? "font-semibold text-foreground" : "text-foreground/70"}`}>{k}</div>
                          <div className="flex-1 h-2 bg-muted rounded">
                            <div className="h-2 bg-accent rounded" style={{ width: `${Math.min(100, Math.max(0, v * 100)).toFixed(0)}%` }} />
                          </div>
                          <div className="w-12 text-right text-xs tabular-nums">{(v * 100).toFixed(0)}%</div>
                        </div>
                      ))}
                    {/* Dominant Emotion Highlight */}
                    {(() => {
                      const [k, v] = Object.entries(liveEmotion.probs).sort((a,b)=>b[1]-a[1])[0]
                      const em = k === "happy" ? "😊" : k === "sad" ? "😢" : k === "angry" ? "😠" : k === "surprise" ? "😲" : k === "fear" ? "😨" : k === "disgust" ? "🤢" : "😐"
                      return (
                        <div className="mt-4 pt-4 border-t border-border space-y-3">
                          <div className="rounded-lg bg-background border border-border p-3 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <span className="text-3xl leading-none">{em}</span>
                              <div className="flex flex-col">
                                <span className="text-xs text-foreground/70 uppercase tracking-wide">Detected</span>
                                <span className="text-base font-bold capitalize text-foreground">{k}</span>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-accent">{(v * 100).toFixed(0)}%</div>
                              <div className="text-[10px] text-foreground/70 uppercase">Confidence</div>
                            </div>
                          </div>
                          {/* Age, Gender, Race Info */}
                          <div className="grid grid-cols-3 gap-2">
                            {liveEmotion.age && liveEmotion.age > 0 && (
                              <div className="rounded-lg border border-border bg-background/80 px-3 py-2">
                                <span className="text-[10px] uppercase tracking-wide text-foreground/60 block">Age</span>
                                <span className="block text-sm font-semibold text-foreground">{liveEmotion.age}y</span>
                              </div>
                            )}
                            {liveEmotion.gender && (
                              <div className="rounded-lg border border-border bg-background/80 px-3 py-2">
                                <span className="text-[10px] uppercase tracking-wide text-foreground/60 block">Gender</span>
                                <span className="block text-sm font-semibold text-foreground capitalize">
                                  {liveEmotion.gender === "Man" ? "♂ Man" : liveEmotion.gender === "Woman" ? "♀ Woman" : liveEmotion.gender}
                                </span>
                              </div>
                            )}
                            {liveEmotion.race && (
                              <div className="rounded-lg border border-border bg-background/80 px-3 py-2">
                                <span className="text-[10px] uppercase tracking-wide text-foreground/60 block">Race</span>
                                <span className="block text-sm font-semibold text-foreground capitalize">{liveEmotion.race}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )
                    })()}
                  </div>
                ) : (
                  <div className="text-xs text-foreground/70 flex-1 flex items-center justify-center text-center px-3">
                    No emotion yet. Capture to view.
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="rounded-lg border border-border p-3 space-y-3 flex-shrink-0">
                <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  <Button variant="secondary" onClick={handleVerify} className="w-full gap-2 h-10 rounded-lg shadow-sm" disabled={isCapturing || (mode === "upload" && !selectedImage)}>
                    <Camera className="h-4 w-4 stroke-[1.5]" />
                    {isCapturing ? "Verifying..." : mode === "webcam" && !selectedImage ? "Capture & Verify" : "Verify Face"}
                  </Button>
                </motion.div>
                {(selectedImage || previewUrl) && (
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button
                      variant="outline"
                      onClick={handleReset}
                      className="w-full gap-2 h-10 rounded-lg"
                    >
                      <RotateCcw className="h-4 w-4 stroke-[1.5]" />
                      Reset
                    </Button>
                  </motion.div>
                )}
                {mode === "webcam" ? (
                  <div className="grid grid-cols-1 gap-2">
                    <Button
                      variant="outline"
                      className={`w-full gap-2 h-10 rounded-lg transition-colors ${isAutoCapture ? "bg-accent text-accent-foreground hover:bg-accent/90 border-transparent" : ""}`}
                      onClick={() => setIsAutoCapture(!isAutoCapture)}
                    >
                      <RefreshCw className={`h-4 w-4 stroke-[1.5] ${isAutoCapture ? "animate-spin" : ""}`} />
                      Auto-capture
                    </Button>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* (Right column includes all controls; removed duplicate buttons) */}
        </CardContent>
      </Card>

      {/* Spoof Detection Dialog */}
      <Dialog open={spoofDialogOpen} onOpenChange={setSpoofDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-full bg-destructive/10">
                <ShieldAlert className="h-6 w-6 text-destructive" />
              </div>
              <DialogTitle className="text-lg font-bold">Spoof Detected</DialogTitle>
            </div>
            <DialogDescription className="text-sm text-foreground/70 pt-2">
              {spoofMessage || "Anti-spoof detection has identified that the image may not be from a real face."}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <p className="text-sm text-foreground/80 mb-4">
              For security reasons, verification has been blocked. Please use a real face (not a photo or screen) to verify.
            </p>
            <div className="flex flex-col gap-2 text-xs text-foreground/60">
              <p className="font-medium">Tips:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Use your actual face, not a photo</li>
                <li>Ensure good lighting</li>
                <li>Look directly at the camera</li>
                <li>Remove any masks or coverings</li>
              </ul>
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setSpoofDialogOpen(false)}
            >
              Understood
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  )
}
