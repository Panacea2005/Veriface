"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import { Camera, X, RotateCcw, CheckCircle2, AlertCircle, ArrowLeft, ArrowRight, ArrowUp, ArrowDown, User } from "lucide-react"
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

type CaptureAngle = "front" | "left" | "right" | "up" | "down"

interface CaptureStep {
  angle: CaptureAngle
  label: string
  icon: React.ReactNode
  instruction: string
  completed: boolean
  image?: string
}

const CAPTURE_STEPS: Omit<CaptureStep, "completed" | "image">[] = [
  {
    angle: "front",
    label: "Front",
    icon: <User className="h-6 w-6" />,
    instruction: "Look straight at the camera"
  },
  {
    angle: "left",
    label: "Left",
    icon: <ArrowLeft className="h-6 w-6" />,
    instruction: "Turn your face to the left"
  },
  {
    angle: "right",
    label: "Right",
    icon: <ArrowRight className="h-6 w-6" />,
    instruction: "Turn your face to the right"
  },
  {
    angle: "up",
    label: "Up",
    icon: <ArrowUp className="h-6 w-6" />,
    instruction: "Tilt your head up"
  },
  {
    angle: "down",
    label: "Down",
    icon: <ArrowDown className="h-6 w-6" />,
    instruction: "Tilt your head down"
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
  const [autoCapture, setAutoCapture] = useState(false)
  const [nameError, setNameError] = useState<string | null>(null)
  const [isCheckingName, setIsCheckingName] = useState(false)
  
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
      
      // Move to next step
      if (currentStep < steps.length - 1) {
        setTimeout(() => {
          setCurrentStep(prev => prev + 1)
          setIsProcessing(false)
        }, 500)
      } else {
        setIsProcessing(false)
      }
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

  // Auto-capture logic using simple timer
  useEffect(() => {
    if (!autoCapture || !open || !videoReady) return
    
    const checkAndCapture = () => {
      // Don't auto-capture if current step is already completed
      if (steps[currentStep].completed) return
      
      // Don't trigger if already capturing/processing
      if (isCapturing || isProcessing) return
      
      // Check if enough time has passed since last capture
      const now = Date.now()
      const timeSinceLastCapture = now - lastCaptureTimeRef.current
      
      // For first capture, wait 2 seconds. For subsequent, wait 3 seconds (allows time to move to next angle)
      const minWaitTime = lastCaptureTimeRef.current === 0 ? 2000 : 3000
      
      if (timeSinceLastCapture >= minWaitTime) {
        // Trigger auto-capture
        console.log(`[Auto-capture] Triggering for ${steps[currentStep].label}`)
        handleCapture()
      }
    }
    
    // Check every 500ms
    const interval = setInterval(checkAndCapture, 500)
    
    return () => {
      clearInterval(interval)
    }
  }, [autoCapture, open, videoReady, currentStep, isCapturing, isProcessing, steps, handleCapture])

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
      setAutoCapture(false)
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
        return
      }
      setNameError(null)
    } catch (error) {
      console.error("[Submit] Name check error:", error)
      // Continue with registration if check fails
    } finally {
      setIsCheckingName(false)
    }

    const completedSteps = steps.filter(s => s.completed && s.image)
    if (completedSteps.length === 0) {
      toast({
        title: "No images captured",
        description: "Please capture at least one image",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)
    try {
      // Convert base64 images to Blobs
      const imageFiles: File[] = []
      for (const step of completedSteps) {
        if (!step.image) continue
        
        // Convert base64 to Blob
        const response = await fetch(step.image)
        const blob = await response.blob()
        const file = new File([blob], `${name}_${step.angle}.jpg`, { type: 'image/jpeg' })
        imageFiles.push(file)
      }
      
      // Register all images in batch (each angle saved as separate embedding)
      const result = await registerFaceBatch(name.trim(), imageFiles)

      const embeddingsCount = result.embeddings_saved || result.images_processed || completedSteps.length
      toast({
        title: "Registration successful",
        description: `Registered ${embeddingsCount} angle${embeddingsCount > 1 ? 's' : ''} for ${name}. Each angle saved as a separate embedding for better accuracy.`,
      })
      
      // Trigger registry update
      window.dispatchEvent(new Event('registry-updated'))
      
      // Close dialog and reset
      onOpenChange(false)
      setName("")
      setCurrentStep(0)
      setSteps(CAPTURE_STEPS.map(s => ({ ...s, completed: false })))
    } catch (error) {
      toast({
        title: "Registration failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
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
              <DialogTitle className="text-base">Register Face - Multi-Angle Capture</DialogTitle>
              <DialogDescription className="text-[11px]">
                Capture your face from multiple angles for better accuracy
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

                {/* Auto-capture Toggle - Top Left */}
                <div className="absolute top-2 left-2 z-10">
                  <motion.div
                    className="bg-background/95 backdrop-blur-sm rounded-lg px-3 py-2 border border-border shadow-lg"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoCapture}
                        onChange={(e) => setAutoCapture(e.target.checked)}
                        className="w-4 h-4 rounded border-border accent-accent"
                      />
                      <span className="text-xs font-medium">Auto-capture</span>
                    </label>
                  </motion.div>
                </div>

                {/* Capture Button - Bottom Center */}
                <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 z-10">
                  <Button
                    onClick={handleCapture}
                    size="lg"
                    className="gap-2 shadow-lg"
                    disabled={!videoReady || isCapturing || isProcessing || steps[currentStep].completed || autoCapture}
                  >
                    {isCapturing || isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-current border-t-transparent" />
                        {autoCapture ? "Auto-capturing..." : "Capturing..."}
                      </>
                    ) : autoCapture ? (
                      <>
                        <Camera className="h-5 w-5" />
                        Auto Mode Active
                      </>
                    ) : (
                      <>
                        <Camera className="h-5 w-5" />
                        Capture {currentStepData.label}
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
              <Label className="text-xs font-medium mb-2 flex-shrink-0">Captured Angles</Label>
              <div className="relative flex-1 min-h-0 overflow-visible">
                {/* 2D Stack of Images - Each image overlaps bottom-right corner of previous, positioned at top-left */}
                {steps.map((step, index) => {
                  const totalSteps = steps.length
                  // Front (index 0) is on top, last image is at bottom
                  const stackIndex = index // 0 = top (front), 4 = bottom
                  // Each image shifts down and to the right, overlapping bottom-right corner
                  // All images start from top-left corner
                  const baseTranslateY = stackIndex * 50 // 0px, 50px, 100px, 150px, 200px
                  const baseTranslateX = stackIndex * 80 // 0px, 80px, 160px, 240px, 320px (wider spacing on x-axis)
                  const translateY = baseTranslateY
                  const translateX = baseTranslateX
                  // Fix z-index: Front (index 0) should have HIGHEST z-index
                  const zIndex = totalSteps + 10 - stackIndex // Front: 15, Last: 11 (higher = on top)
                  
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
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end">
                            <div className="p-2 w-full">
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-medium text-white">
                                  {step.label}
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
                })}
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

