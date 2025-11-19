"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import { Camera, X, RotateCcw, CheckCircle2, AlertCircle, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { motion, AnimatePresence } from "framer-motion"
import { registerFaceBatch, checkNameExists } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

type CaptureAngle = "front" | "processing"

interface CaptureStep {
  angle: CaptureAngle
  label: string
  icon: React.ReactNode
  instruction: string
  completed: boolean
  image?: string
  processingResult?: {
    status: "processing" | "processed" | "skipped" | "error"
    stage?: string
    message?: string
    torch_norm?: number
    embedding_dim?: number
  }
}

const CAPTURE_STEPS: Omit<CaptureStep, "completed" | "image">[] = [
  {
    angle: "front",
    label: "Front",
    icon: <User className="h-6 w-6" />,
    instruction: "Look straight at the camera"
  }
]

// Processing stages for visualization
const PROCESSING_STAGES: Omit<CaptureStep, "completed" | "image">[] = [
  {
    angle: "processing",
    label: "Stage 1",
    icon: <Camera className="h-6 w-6" />,
    instruction: "Face Detection & Alignment"
  },
  {
    angle: "processing",
    label: "Stage 2",
    icon: <CheckCircle2 className="h-6 w-6" />,
    instruction: "PyTorch Model Processing"
  },
  {
    angle: "processing",
    label: "Stage 3",
    icon: <CheckCircle2 className="h-6 w-6" />,
    instruction: "Feature Extraction"
  },
  {
    angle: "processing",
    label: "Stage 4",
    icon: <CheckCircle2 className="h-6 w-6" />,
    instruction: "Embedding Extraction"
  }
]

interface RegisterWebcamDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function RegisterWebcamDialog({ open, onOpenChange }: RegisterWebcamDialogProps) {
  const [name, setName] = useState("")
  const [currentStep, setCurrentStep] = useState(0)
  const [steps, setSteps] = useState<CaptureStep[]>(
    CAPTURE_STEPS.map(s => ({ ...s, completed: false }))
  )
  const [isCapturing, setIsCapturing] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [videoReady, setVideoReady] = useState(false)
  const [nameError, setNameError] = useState<string | null>(null)
  const [isCheckingName, setIsCheckingName] = useState(false)
  const [processingStages, setProcessingStages] = useState<CaptureStep[]>([])
  const [currentProcessingStage, setCurrentProcessingStage] = useState<number>(-1)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const lastCaptureTimeRef = useRef<number>(0)
  
  const { toast } = useToast()

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
          // Check if video is ready
          if (video.readyState >= 2) {
            setVideoReady(true)
          }
          video.removeEventListener('loadedmetadata', onLoaded)
        }
        const onCanPlay = () => {
          setVideoReady(true)
          video.removeEventListener('canplay', onCanPlay)
        }
        video.addEventListener('loadedmetadata', onLoaded)
        video.addEventListener('canplay', onCanPlay)
        try { 
          await video.play()
          // Double check after play
          if (video.readyState >= 2) {
            setVideoReady(true)
          }
        } catch {}
      }
    } catch (error) {
      toast({
        title: "Camera access denied",
        description: "Please allow camera access to register",
        variant: "destructive",
      })
      onOpenChange(false)
    }
  }, [toast, onOpenChange])

  const captureImage = useCallback(async (): Promise<string | null> => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA) return null

    // Crop the webcam frame to a centered square to reduce background noise
    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const squareSize = Math.min(videoWidth, videoHeight)
    const offsetX = (videoWidth - squareSize) / 2
    const offsetY = (videoHeight - squareSize) / 2

    canvas.width = squareSize
    canvas.height = squareSize
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    // Flip horizontal for mirror effect, then draw the centered square crop
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
    return canvas.toDataURL('image/jpeg', 0.95)
  }, [])

  const handleSubmitAfterCapture = useCallback(async (imageData: string) => {
    // Check if name already exists
    setIsCheckingName(true)
    try {
      const nameExists = await checkNameExists(name.trim())
      if (nameExists) {
        setNameError("This name is already registered. Please use a different name.")
        toast({
          title: "Name already exists",
          description: "This name is already registered. Please choose a different name.",
          variant: "destructive",
        })
        setIsProcessing(false)
        return
      }
      setNameError(null)
    } catch (error) {
      console.error("[Submit] Name check error:", error)
      // Continue with registration if check fails
    } finally {
      setIsCheckingName(false)
    }

    setIsSubmitting(true)
    
    // Convert base64 image to Blob
    const response = await fetch(imageData)
    const blob = await response.blob()
    const file = new File([blob], `${name}_front.jpg`, { type: 'image/jpeg' })
    
    // Initialize processing stages for visualization
    const stages = PROCESSING_STAGES.map((stage, idx) => ({
      ...stage,
      completed: false,
      image: imageData, // Use captured image for all stages
      processingResult: {
        status: "processing" as const,
        stage: stage.label,
        message: stage.instruction
      }
    }))
    setProcessingStages(stages)
    setCurrentProcessingStage(0)

    try {
      // Start registration API call in parallel with animations
      const registerPromise = registerFaceBatch(name.trim(), [file])
      
      // Stage 1: Face Detection & Alignment (400ms)
      await new Promise(resolve => setTimeout(resolve, 400))
      setProcessingStages(prev => {
        const updated = [...prev]
        updated[0] = {
          ...updated[0],
          completed: true,
          processingResult: {
            status: "processed",
            stage: "Stage 1",
            message: "Face detected and aligned"
          }
        }
        return updated
      })
      setCurrentProcessingStage(1)

      // Stage 2: PyTorch Model Processing (500ms)
      await new Promise(resolve => setTimeout(resolve, 500))
      setProcessingStages(prev => {
        const updated = [...prev]
        updated[1] = {
          ...updated[1],
          completed: true,
          processingResult: {
            status: "processed",
            stage: "Stage 2",
            message: "PyTorch embedding extracted",
            torch_norm: 1.0,
            embedding_dim: 512
          }
        }
        return updated
      })
      setCurrentProcessingStage(2)

      // Stage 3: Feature Extraction (500ms)
      await new Promise(resolve => setTimeout(resolve, 500))
      setProcessingStages(prev => {
        const updated = [...prev]
        updated[2] = {
          ...updated[2],
          completed: true,
          processingResult: {
            status: "processed",
            stage: "Stage 3",
            message: "Feature extraction complete",
            embedding_dim: 512
          }
        }
        return updated
      })
      setCurrentProcessingStage(3)

      // Wait for registration to complete (API call should finish around this time)
      const result = await registerPromise

      // Stage 4: Embedding Extraction & Normalization (use real data from API)
      // Small delay to show transition to embedding numbers
      await new Promise(resolve => setTimeout(resolve, 300))
      const procResult = result.processing_results?.[0]
      setProcessingStages(prev => {
        const updated = [...prev]
        updated[3] = {
          ...updated[3],
          completed: true,
          processingResult: {
            status: "processed",
            stage: "Stage 4",
            message: "Embeddings normalized and saved",
            torch_norm: procResult?.torch_norm || 1.0,
            embedding_dim: 512
          }
        }
        return updated
      })

      // Update main step with final result
      if (result.processing_results && result.processing_results.length > 0) {
        const finalProcResult = result.processing_results[0]
        setSteps(prevSteps => {
          const updated = [...prevSteps]
          updated[0] = {
            ...updated[0],
            processingResult: {
              status: "processed",
              torch_norm: finalProcResult.torch_norm
            }
          }
          return updated
        })
      }

      // Wait for Stage 4 animation to complete
      // Stage 4 animation: 0.3s delay + 0.2s transition + 512 dots animation (~2s) = ~2.5s
      // Add extra buffer for smooth completion
      await new Promise(resolve => setTimeout(resolve, 2500))

      toast({
        title: "Registration successful",
        description: `Front-facing image registered for ${name}. Processing complete!`,
      })
      
      // Trigger registry update
      window.dispatchEvent(new Event('registry-updated'))
      
      // Close dialog and reset after showing success message
      setTimeout(() => {
        onOpenChange(false)
        setName("")
        setCurrentStep(0)
        setSteps(CAPTURE_STEPS.map(s => ({ ...s, completed: false })))
        setProcessingStages([])
        setCurrentProcessingStage(-1)
      }, 2500)
    } catch (error) {
      toast({
        title: "Registration failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
      setIsProcessing(false)
      setProcessingStages([])
      setCurrentProcessingStage(-1)
    } finally {
      setIsSubmitting(false)
    }
  }, [name, toast, onOpenChange])

  const handleCapture = useCallback(async () => {
    // Prevent multiple captures
    if (isCapturing || isProcessing) {
      return
    }

    setIsCapturing(true)
    setIsProcessing(true)
    const captureTime = Date.now()
    lastCaptureTimeRef.current = captureTime // Update last capture time
    
    try {
      const imageData = await captureImage()
      if (!imageData) {
        throw new Error("Failed to capture image")
      }
      
      // Debug: Log capture info
      const currentAngle = steps[currentStep].angle
      console.log(`[DEBUG] Capture: angle=${currentAngle}, step=${currentStep + 1}, time=${captureTime}, dataLength=${imageData.length}`)
      
      // Update step with captured image
      setSteps(prev => {
        const newSteps = [...prev]
        newSteps[currentStep].completed = true
        newSteps[currentStep].image = imageData
        return newSteps
      })
      
      // Just capture, don't auto-submit (user will click Register button)
          setIsProcessing(false)
    } catch (error) {
      console.error("[Capture] Error:", error)
      toast({
        title: "Capture failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
      setIsProcessing(false)
    } finally {
      // Reset capturing state after a short delay to allow UI to update
      setTimeout(() => {
        setIsCapturing(false)
      }, 100)
    }
  }, [currentStep, steps.length, captureImage, toast, isCapturing, isProcessing])

  // Auto-capture removed - only manual capture for single front image

  useEffect(() => {
    if (open) {
      startWebcam()
    } else {
      stopWebcam()
      // Reset state when closing
      setName("")
      setCurrentStep(0)
      setSteps(CAPTURE_STEPS.map(s => ({ ...s, completed: false })))
      setIsCapturing(false)
      setIsProcessing(false)
      setVideoReady(false)
      setNameError(null)
      setIsCheckingName(false)
      lastCaptureTimeRef.current = 0
    }
    
    return () => {
      stopWebcam()
    }
  }, [open, startWebcam, stopWebcam])

  const handleRetry = (stepIndex: number) => {
    const newSteps = [...steps]
    newSteps[stepIndex].completed = false
    newSteps[stepIndex].image = undefined
    setSteps(newSteps)
    setCurrentStep(stepIndex)
  }

  const handleSubmit = async () => {
    if (!name.trim()) {
      toast({
        title: "Name required",
        description: "Please enter your name",
        variant: "destructive",
      })
      return
    }

    const completedSteps = steps.filter(s => s.completed && s.image)
    if (completedSteps.length === 0) {
      toast({
        title: "No images captured",
        description: "Please capture an image first",
        variant: "destructive",
      })
      return
    }

    // Use the same logic as handleSubmitAfterCapture
    if (completedSteps[0]?.image) {
      await handleSubmitAfterCapture(completedSteps[0].image)
    }
  }

  const currentStepData = steps[currentStep]
  const completedCount = steps.filter(s => s.completed).length
  const progress = (completedCount / steps.length) * 100

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[98vw] max-w-[98vw] md:max-w-[98vw] lg:max-w-[98vw] xl:max-w-[98vw] p-0 overflow-hidden max-h-[98vh] flex flex-col">
      <DialogHeader className="px-4 pt-2 pb-1.5 flex-shrink-0">
          <div className="flex items-start justify-between gap-3">
            <div>
              <DialogTitle className="text-base">Register Face</DialogTitle>
              <DialogDescription className="text-[11px]">
                Capture your front-facing photo for registration
              </DialogDescription>
            </div>
            {/* Model selection removed; always using Model A */}
          </div>
        </DialogHeader>
        
        <div className="flex-1 overflow-hidden px-4 pb-3 flex flex-col min-h-0">
          {/* Name Input and Progress - Compact layout */}
          <div className="flex justify-end mb-1.5 flex-shrink-0">
            <div className="w-full md:w-auto md:min-w-[240px] space-y-1.5">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Label htmlFor="name" className="text-xs font-medium whitespace-nowrap">
                    Name:
                  </Label>
                  <Input
                    id="name"
                    placeholder="Enter name"
                    value={name}
                    onChange={(e) => {
                      setName(e.target.value)
                      setNameError(null) // Clear error when user types
                    }}
                    className={`rounded-lg border-border h-8 text-sm flex-1 ${nameError ? 'border-red-500' : ''}`}
                    disabled={isSubmitting || isCheckingName}
                  />
                </div>
                {nameError && (
                  <p className="text-xs text-red-500 pl-12">{nameError}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs font-medium whitespace-nowrap">Progress:</Label>
                <div className="flex-1 flex items-center gap-2">
                  <Progress value={progress} className="h-1.5 flex-1" />
                  <Badge variant="outline" className="text-xs px-1.5 py-0 h-5">
                    {completedCount}/{steps.length}
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Main Capture Area */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-3 flex-1 min-h-0">
            {/* Webcam Preview - Left */}
            <div className="lg:col-span-3 flex items-start justify-center min-h-0">
              <div className="relative aspect-square rounded-lg border-2 border-border bg-black overflow-hidden w-full max-w-[480px]">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                  style={{ transform: "scaleX(-1)" }}
                />
                <canvas 
                  ref={canvasRef} 
                  className="hidden"
                />

                {/* Current Step Overlay - Small at top right */}
                <div className="absolute top-2 right-2 pointer-events-none z-10">
                  <motion.div
                    className="bg-background/95 backdrop-blur-sm rounded-lg px-3 py-2 border border-border shadow-lg"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    key={currentStep}
                  >
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-full bg-accent/20 text-accent">
                        {currentStepData.icon}
                      </div>
                      <div className="text-left">
                        <h3 className="text-xs font-bold leading-tight">{currentStepData.label}</h3>
                        <p className="text-[10px] text-foreground/70 leading-tight">{currentStepData.instruction}</p>
                      </div>
                    </div>
                  </motion.div>
                </div>

                {/* Auto-capture removed - only manual capture for single front image */}

                {/* Capture Button - Bottom Center */}
                <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 z-10">
                  <Button
                    onClick={handleCapture}
                    size="lg"
                    className="gap-2 shadow-lg"
                    disabled={!videoReady || isCapturing || isProcessing || steps[currentStep].completed}
                  >
                    {isCapturing || isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-current border-t-transparent" />
                        {isProcessing ? "Processing..." : "Capturing..."}
                      </>
                    ) : (
                      <>
                        <Camera className="h-5 w-5" />
                        Capture Front
                      </>
                    )}
                  </Button>
                </div>

                {/* Processing Overlay */}
                {isProcessing && (
                  <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center">
                    <motion.div
                      className="text-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      <div className="animate-spin rounded-full h-12 w-12 border-4 border-accent border-t-transparent mx-auto mb-2" />
                      <p className="text-sm font-medium">Processing...</p>
                    </motion.div>
                  </div>
                )}
              </div>
            </div>

            {/* Preview Grid - Right */}
            <div className="lg:col-span-2 flex flex-col min-h-0">
              <Label className="text-xs font-medium mb-2 flex-shrink-0">
                {isSubmitting ? "Processing Stages (5 frames)" : "Captured Image"}
              </Label>
              <div className="relative flex-1 min-h-0 overflow-visible">
                {/* Show 5 frames: 1 captured image + 4 processing stages */}
                {isSubmitting && processingStages.length > 0 ? (
                  // 5 frames: captured image (index 0) + 4 processing stages (index 1-4)
                  [
                    // Frame 0: Captured image
                    ...steps.filter(s => s.completed && s.image).map((step, stepIdx) => ({
                      ...step,
                      index: 0,
                      isCaptured: true
                    })),
                    // Frames 1-4: Processing stages
                    ...processingStages.map((stage, stageIdx) => ({
                      ...stage,
                      index: stageIdx + 1,
                      isCaptured: false
                    }))
                  ].map((item, index) => {
                    const stage = item.isCaptured ? null : item as CaptureStep
                    const step = item.isCaptured ? item as CaptureStep : null
                    const displayIndex = item.index
                    const totalFrames = 5 // 1 captured + 4 stages
                    const stackIndex = displayIndex
                    const baseTranslateY = stackIndex * 50
                    const baseTranslateX = stackIndex * 80
                    const translateY = baseTranslateY
                    const translateX = baseTranslateX
                    const zIndex = totalFrames + 10 - stackIndex
                    
                    // For captured image (index 0)
                    if (item.isCaptured && step) {
                      return (
                        <motion.div
                          key="captured-image"
                          className="absolute top-0 left-0 aspect-square rounded-md border-2 overflow-hidden border-accent bg-background"
                          style={{
                            zIndex: hoveredIndex === 0 ? 999 : zIndex,
                            width: "60%",
                            height: "60%",
                            maxWidth: "200px",
                            maxHeight: "200px",
                          }}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ 
                            opacity: 1,
                            scale: hoveredIndex === 0 ? 1.1 : 1,
                            y: hoveredIndex === 0 ? translateY - 20 : translateY,
                            x: hoveredIndex === 0 ? translateX - 40 : translateX,
                          }}
                          onHoverStart={() => setHoveredIndex(0)}
                          onHoverEnd={() => setHoveredIndex(null)}
                          transition={{ duration: 0.4, ease: "easeOut" }}
                        >
                          {step.image && (
                            <>
                              <img
                                src={step.image}
                                alt="Captured"
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent flex flex-col justify-between">
                                <div className="p-2 w-full">
                                  <span className="text-xs font-bold text-white">Captured</span>
                                </div>
                              </div>
                            </>
                          )}
                        </motion.div>
                      )
                    }
                    
                    // For processing stages (index 1-4)
                    if (!item.isCaptured && stage) {
                      const stageIndex = displayIndex - 1 // Convert to 0-3 for processing stages
                      const isActive = currentProcessingStage === stageIndex
                      const isCompleted = stage.completed
                      const isProcessing = stage.processingResult?.status === "processing"
                      
                      // Progressive visualization matching actual embedding process:
                      // Stage 1: Face Detection & Alignment - show face detection overlay
                      // Stage 2: PyTorch Model - show embedding vector being generated (first 128 dims)
                      // Stage 3: Feature Extraction - show embedding vector being generated (next 128 dims)
                      // Stage 4: Normalization & Save - show complete normalized 512-D embedding vector
                      const showFaceDetection = stageIndex === 0 && (isActive || isProcessing) // Stage 1: face detection
                      const showPyTorchEmbedding = stageIndex === 1 && (isActive || isCompleted) // Stage 2: PyTorch embedding (first 128 dims)
                      const showFeatureExtraction = stageIndex === 2 && (isActive || isCompleted) // Stage 3: Feature extraction (next 128 dims)
                      const showFinalEmbedding = stageIndex === 3 && isCompleted // Stage 4: final normalized embedding (all 512 dims)
                      const imageOpacity = stageIndex === 0 ? 1 : Math.max(0, 1 - (stageIndex * 0.25)) // Fade out image gradually
                      const embeddingOpacity = (showPyTorchEmbedding || showFeatureExtraction || showFinalEmbedding) ? 1 : 0
                      
                      return (
                        <motion.div
                          key={`stage-${displayIndex}`}
                          className={`absolute top-0 left-0 aspect-square rounded-md border-2 overflow-hidden ${
                            isCompleted
                              ? "border-green-500 bg-background"
                              : isActive || isProcessing
                              ? "border-blue-500 bg-background"
                              : "border-border bg-muted opacity-50"
                          }`}
                          style={{
                            zIndex: hoveredIndex === displayIndex ? 999 : zIndex,
                            width: "60%",
                            height: "60%",
                            maxWidth: "200px",
                            maxHeight: "200px",
                          }}
                          initial={{ opacity: 0, scale: 0.8, y: translateY + 20, x: translateX + 20 }}
                          animate={{ 
                            opacity: isActive || isCompleted ? 1 : 0.5,
                            scale: hoveredIndex === displayIndex ? 1.1 : (isActive ? 1.05 : 1),
                            y: hoveredIndex === displayIndex ? translateY - 20 : translateY,
                            x: hoveredIndex === displayIndex ? translateX - 40 : translateX,
                          }}
                          onHoverStart={() => setHoveredIndex(displayIndex)}
                          onHoverEnd={() => setHoveredIndex(null)}
                          transition={{ duration: 0.4, ease: "easeOut" }}
                        >
                        {stage.image ? (
                          <>
                            {/* Base image - fades out as we progress through stages */}
                            <motion.div
                              className="absolute inset-0"
                              animate={{
                                opacity: imageOpacity,
                                filter: stageIndex > 0 ? "grayscale(100%) brightness(0.2)" : "grayscale(0%) brightness(1)",
                              }}
                              transition={{ duration: 0.8, ease: "easeInOut" }}
                            >
                              <img
                                src={stage.image}
                                alt={stage.label}
                                className="w-full h-full object-cover"
                              />
                            </motion.div>
                            
                            {/* Face Detection Overlay (Stage 1) */}
                            {showFaceDetection && (
                              <motion.div
                                className="absolute inset-0 pointer-events-none"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.3 }}
                              >
                                {/* Face bounding box overlay */}
                                <motion.div
                                  className="absolute top-1/4 left-1/4 w-1/2 h-1/2 border-[3px] border-black bg-transparent"
                                  style={{
                                    boxShadow: "4px 4px 0 0 black"
                                  }}
                                  initial={{ scale: 0.8, opacity: 0 }}
                                  animate={{ scale: 1, opacity: 1 }}
                                  transition={{ duration: 0.4, ease: "easeOut" }}
                                />
                                {/* Face landmarks (simplified - 5 points) */}
                                {[
                                  { x: "35%", y: "40%" }, // Left eye
                                  { x: "65%", y: "40%" }, // Right eye
                                  { x: "50%", y: "55%" }, // Nose
                                  { x: "40%", y: "70%" }, // Left mouth
                                  { x: "60%", y: "70%" }, // Right mouth
                                ].map((point, idx) => (
                                  <motion.div
                                    key={idx}
                                    className="absolute w-2 h-2 bg-black border-[2px] border-black"
                                    style={{
                                      left: point.x,
                                      top: point.y,
                                      transform: "translate(-50%, -50%)",
                                      boxShadow: "2px 2px 0 0 black"
                                    }}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ delay: 0.2 + idx * 0.1, duration: 0.2 }}
                                  />
                                ))}
                              </motion.div>
                            )}

                            {/* Embedding Vector Visualization (Stages 2-4) - Neo Brutalism Style */}
                            {(showPyTorchEmbedding || showFeatureExtraction || showFinalEmbedding) && (
                              <motion.div
                                className="absolute inset-0 bg-white"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: embeddingOpacity }}
                                transition={{ duration: 0.5, ease: "easeOut" }}
                              >
                                {(() => {
                                  // Determine grid layout and dot size based on stage
                                  let totalDots = 0
                                  let cols = 16
                                  let rows = 0
                                  let gap = 1
                                  
                                  if (showFinalEmbedding) {
                                    // Stage 4: 512 dots - 16x32 grid, smaller dots
                                    totalDots = 512
                                    cols = 16
                                    rows = 32
                                    gap = 0.5
                                  } else if (showFeatureExtraction) {
                                    // Stage 3: 256 dots - 16x16 grid, medium dots
                                    totalDots = 256
                                    cols = 16
                                    rows = 16
                                    gap = 1
                                  } else if (showPyTorchEmbedding) {
                                    // Stage 2: 128 dots - 16x8 grid, larger dots
                                    totalDots = 128
                                    cols = 16
                                    rows = 8
                                    gap = 1.5
                                  }
                                  
                                  return (
                                    <motion.div 
                                      className="w-full h-full p-2 flex items-center justify-center"
                                      initial={{ opacity: 0 }}
                                      animate={{ opacity: embeddingOpacity }}
                                      transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
                                    >
                                      <div 
                                        className="grid w-full h-full"
                                        style={{
                                          gridTemplateColumns: `repeat(${cols}, 1fr)`,
                                          gridTemplateRows: `repeat(${rows}, 1fr)`,
                                          gap: `${gap * 4}px`
                                        }}
                                      >
                                        {Array.from({ length: totalDots }).map((_, numIdx) => {
                                          // Batch delays: group dots into batches for smoother animation
                                          const batchSize = 4
                                          const batchIdx = Math.floor(numIdx / batchSize)
                                          const withinBatchIdx = numIdx % batchSize
                                          const batchDelay = batchIdx * 0.008 // 8ms per batch
                                          const withinBatchDelay = withinBatchIdx * 0.002 // 2ms within batch
                                          const totalDelay = batchDelay + withinBatchDelay
                                          
                                          return (
                                            <motion.div
                                              key={numIdx}
                                              className="w-full h-full"
                                              initial={{ opacity: 0, scale: 0 }}
                                              animate={{ 
                                                opacity: 1, 
                                                scale: 1 
                                              }}
                                              transition={{
                                                duration: 0.2,
                                                delay: totalDelay,
                                                ease: "easeOut"
                                              }}
                                              style={{ willChange: "transform, opacity" }}
                                            >
                                              <div 
                                                className="w-full h-full bg-black border-[2px] border-black"
                                                style={{
                                                  boxShadow: "2px 2px 0 0 black"
                                                }}
                                              />
                                            </motion.div>
                                          )
                                        })}
                                      </div>
                                    </motion.div>
                                  )
                                })()}
                              </motion.div>
                            )}
                          </>
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-muted">
                            <div className="text-center">
                              <div className="p-2 rounded-full bg-accent/20 text-accent mb-2 mx-auto w-fit">
                                {stage.icon}
                              </div>
                              <p className="text-xs font-medium">{stage.label}</p>
                              <p className="text-[10px] text-muted-foreground">{stage.instruction}</p>
                            </div>
                          </div>
                        )}
                      </motion.div>
                      )
                    }
                    
                    // Fallback (should not happen)
                    return null
                  }).filter(Boolean)
                ) : (
                  // Show captured image when not processing
                  steps.map((step, index) => {
                  const totalSteps = steps.length
                    const stackIndex = index
                    const baseTranslateY = stackIndex * 50
                    const baseTranslateX = stackIndex * 80
                  const translateY = baseTranslateY
                  const translateX = baseTranslateX
                    const zIndex = totalSteps + 10 - stackIndex
                  
                  return (
                    <motion.div
                      key={step.angle}
                      className={`absolute top-0 left-0 aspect-square rounded-md border-2 overflow-hidden cursor-pointer ${
                        step.completed
                          ? "border-accent bg-background"
                          : index === currentStep
                          ? "border-primary bg-background"
                          : "border-border bg-muted"
                      }`}
                      style={{
                        zIndex: hoveredIndex === index ? 999 : zIndex,
                        width: "60%",
                        height: "60%",
                        maxWidth: "200px",
                        maxHeight: "200px",
                      }}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ 
                        opacity: 1, 
                        scale: hoveredIndex === index ? 1.5 : 1,
                        y: hoveredIndex === index ? translateY - 30 : translateY,
                        x: hoveredIndex === index ? translateX - 60 : translateX,
                      }}
                      onHoverStart={() => setHoveredIndex(index)}
                      onHoverEnd={() => setHoveredIndex(null)}
                      transition={{ duration: 0.2, ease: "easeOut" }}
                    >
                      {step.image ? (
                        <>
                          <img
                            src={step.image}
                            alt={step.label}
                            className="w-full h-full object-cover"
                          />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent flex flex-col justify-between">
                              <div className="p-2 w-full flex items-center justify-between">
                                <span className="text-xs font-medium text-white">
                                  {step.label}
                                  {step.processingResult?.status === "processed" && (
                                    <Badge variant="default" className="ml-1.5 h-4 px-1 text-[9px]">Saved</Badge>
                                  )}
                                </span>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="h-6 w-6 p-0 text-white hover:bg-white/20"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleRetry(index)
                                  }}
                                  disabled={isSubmitting}
                                >
                                  <RotateCcw className="h-3 w-3" />
                                </Button>
                            </div>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center gap-2">
                          <div className={`p-2 rounded-full ${
                            index === currentStep ? "bg-primary/20 text-primary" : "bg-muted text-foreground/40"
                          }`}>
                            {step.icon}
                          </div>
                          <span className={`text-xs font-medium ${
                            index === currentStep ? "text-primary" : "text-foreground/40"
                          }`}>
                            {step.label}
                          </span>
                        </div>
                      )}
                    </motion.div>
                  )
                  })
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex gap-2 pt-3 pb-4 px-4 border-t flex-shrink-0">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!name.trim() || completedCount === 0 || isSubmitting || isCheckingName || !!nameError}
            className="flex-1"
          >
            {isSubmitting || isCheckingName ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent mr-2" />
                {isCheckingName ? "Checking name..." : "Registering..."}
              </>
            ) : (
              `Register ${completedCount} Angle${completedCount !== 1 ? 's' : ''}`
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

