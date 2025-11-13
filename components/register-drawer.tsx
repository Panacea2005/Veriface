"use client"

import { useState } from "react"
import { UserPlus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import { RegisterWebcamDialog } from "@/components/register-webcam-dialog"

export function RegisterDrawer() {
  const [webcamDialogOpen, setWebcamDialogOpen] = useState(false)

  return (
    <>
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button 
          size="sm" 
          variant="outline" 
          className="gap-2"
          onClick={() => setWebcamDialogOpen(true)}
        >
            <UserPlus className="h-4 w-4 stroke-[1.5]" />
            Register
          </Button>
        </motion.div>
      <RegisterWebcamDialog 
        open={webcamDialogOpen} 
        onOpenChange={setWebcamDialogOpen} 
      />
    </>
  )
}
