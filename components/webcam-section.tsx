"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Camera, RefreshCw, Upload, ZoomIn, ZoomOut, RotateCcw, FlipHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RegisterDrawer } from "@/components/register-drawer"
import { motion } from "framer-motion"
import { verifyFace, checkHealth } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import type { VerifyResponse, HealthResponse } from "@/lib/api"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

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
  const [isFlipped, setIsFlipped] = useState(false)
  const [registryInfo, setRegistryInfo] = useState<HealthResponse["registry"] | null>(null)
  
  // Webcam refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  const { toast } = useToast()

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
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: "user"
        } 
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
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

  const captureFromWebcam = () => {
    if (!videoRef.current || !canvasRef.current) return
    
    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.drawImage(video, 0, 0)
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

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedImage(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setZoom(1)
      setPosition({ x: 0, y: 0 })
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
      const result = await verifyFace(selectedImage, "A", "cosine")
      onVerifyResult(result)
      toast({
        title: "Verification complete",
        description: result.matched_id 
          ? `Matched: ${result.matched_id} (${(result.score! * 100).toFixed(1)}%)`
          : "No match found",
      })
    } catch (error) {
      toast({
        title: "Verification failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setIsCapturing(false)
    }
  }

  // Use refs to avoid dependency issues
  const onVerifyResultRef = useRef(onVerifyResult)
  const toastRef = useRef(toast)
  
  useEffect(() => {
    onVerifyResultRef.current = onVerifyResult
    toastRef.current = toast
  }, [onVerifyResult, toast])

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
      
      ctx.drawImage(video, 0, 0)
      
      canvas.toBlob(async (blob) => {
        if (!blob) return
        
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" })
        setIsCapturing(true)
        
        try {
          const result = await verifyFace(file, "A", "cosine")
          onVerifyResultRef.current(result)
          toastRef.current({
            title: "Auto-verification complete",
            description: result.matched_id 
              ? `Matched: ${result.matched_id}`
              : "No match found",
          })
        } catch (error) {
          console.error("Auto-verification failed:", error)
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

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <Card className="flex flex-col overflow-hidden border border-border shadow-sm">
        <CardHeader className="pb-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle className="text-base font-semibold">Face Capture & Verification</CardTitle>
              <CardDescription className="text-xs">Upload image or capture from webcam</CardDescription>
            </div>
            {registryInfo && (
              <div className="flex flex-col items-end gap-1 text-xs">
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${registryInfo.accessible ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                  <span className="text-muted-foreground font-medium">
                    {registryInfo.accessible ? "Registry Active" : "Registry Inaccessible"}
                  </span>
                </div>
                {registryInfo.accessible && (
                  <div className="text-muted-foreground/70">
                    {registryInfo.users_count || 0} user{registryInfo.users_count !== 1 ? 's' : ''} • {registryInfo.total_embeddings || 0} embeddings
                  </div>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          {/* Mode Tabs */}
          <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
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

            <TabsContent value="upload" className="mt-4">
              {/* Upload Mode */}
              <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-muted border border-border h-96">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
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
                      className="w-full h-full object-contain"
                      style={{
                        transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                        transformOrigin: "center center",
                        transition: isDragging ? "none" : "transform 0.1s ease-out"
                      }}
                      draggable={false}
                    />
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-muted-foreground">
                    <Upload className="h-12 w-12 opacity-40 stroke-[1.5]" />
                    <span className="text-sm font-medium">Select image to upload</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-2"
                    >
                      Choose Image
                    </Button>
                  </div>
                )}
                {isCapturing && (
                  <motion.div
                    className="absolute inset-0 bg-primary/5 flex items-center justify-center z-10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 0.5, 0] }}
                    transition={{ duration: 0.6 }}
                  >
                    <span className="text-sm font-medium">Processing...</span>
                  </motion.div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="webcam" className="mt-4">
              {/* Webcam Mode */}
              <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-muted border border-border h-96">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-contain ${selectedImage ? "hidden" : ""}`}
                  style={{
                    transform: isFlipped ? "scaleX(-1)" : "none",
                  }}
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
                  <motion.div
                    className="absolute inset-0 bg-primary/5 flex items-center justify-center z-10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 0.5, 0] }}
                    transition={{ duration: 0.6 }}
                  >
                    <span className="text-sm font-medium">Processing...</span>
                  </motion.div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </TabsContent>
          </Tabs>

          {/* Zoom Controls & Flip Camera */}
          <div className="flex items-center justify-center gap-2">
            {previewUrl && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomOut}
                  disabled={zoom <= 0.5}
                  className="gap-2"
                >
                  <ZoomOut className="h-4 w-4" />
                  Zoom Out
                </Button>
                <span className="text-sm text-muted-foreground min-w-[60px] text-center">
                  {Math.round(zoom * 100)}%
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomIn}
                  disabled={zoom >= 3}
                  className="gap-2"
                >
                  <ZoomIn className="h-4 w-4" />
                  Zoom In
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  className="gap-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </>
            )}
            {mode === "webcam" && !selectedImage && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsFlipped(!isFlipped)}
                className="gap-2 ml-auto"
              >
                <FlipHorizontal className="h-4 w-4" />
                Flip
              </Button>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-3">
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button 
                onClick={handleVerify} 
                className="w-full gap-2 h-10 rounded-lg" 
                disabled={isCapturing || (mode === "upload" && !selectedImage)}
              >
                <Camera className="h-4 w-4 stroke-[1.5]" />
                {isCapturing 
                  ? "Verifying..." 
                  : mode === "webcam" && !selectedImage
                  ? "Capture & Verify"
                  : "Verify Face"
                }
              </Button>
            </motion.div>

            <div className={`flex gap-3 ${mode === "webcam" ? "flex-col" : ""}`}>
              {mode === "webcam" ? (
                <>
                  <motion.div className="flex-1" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <RegisterDrawer />
                  </motion.div>
                  <motion.div className="flex-1" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button
                      variant={isAutoCapture ? "default" : "outline"}
                      className="w-full gap-2 h-10 rounded-lg"
                      onClick={() => setIsAutoCapture(!isAutoCapture)}
                    >
                      <RefreshCw className={`h-4 w-4 stroke-[1.5] ${isAutoCapture ? "animate-spin" : ""}`} />
                      Auto-capture
                    </Button>
                  </motion.div>
                </>
              ) : (
                <RegisterDrawer />
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
