"use client"

import { useState } from "react"
import { Camera, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RegisterDrawer } from "@/components/register-drawer"
import { motion } from "framer-motion"

export function WebcamSection() {
  const [isAutoCapture, setIsAutoCapture] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)

  const handleCapture = () => {
    setIsCapturing(true)
    setTimeout(() => setIsCapturing(false), 1000)
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <Card className="flex flex-col overflow-hidden border border-border shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-base font-semibold">Webcam Feed</CardTitle>
          <CardDescription className="text-xs">Real-time face detection and capture</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          {/* Webcam Placeholder - Full Width */}
          <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-muted border border-border h-96">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <Camera className="h-12 w-12 opacity-40 stroke-[1.5]" />
              <span className="text-sm font-medium">Webcam feed</span>
            </div>
            {isCapturing && (
              <motion.div
                className="absolute inset-0 bg-primary/5"
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 0.5, 0] }}
                transition={{ duration: 0.6 }}
              />
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-3">
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button onClick={handleCapture} className="w-full gap-2 h-10 rounded-lg" disabled={isCapturing}>
                <Camera className="h-4 w-4 stroke-[1.5]" />
                {isCapturing ? "Capturing..." : "Capture & Verify"}
              </Button>
            </motion.div>

            <div className="flex gap-3">
              <RegisterDrawer />
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
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
