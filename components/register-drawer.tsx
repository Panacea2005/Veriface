"use client"

import { useState } from "react"
import { UserPlus } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { motion } from "framer-motion"

export function RegisterDrawer() {
  const [name, setName] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = () => {
    setIsSubmitting(true)
    setTimeout(() => {
      setIsSubmitting(false)
      setName("")
    }, 1000)
  }

  return (
    <Drawer>
      <DrawerTrigger asChild>
        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button variant="outline" className="flex-1 gap-2 rounded-lg h-10 bg-transparent">
            <UserPlus className="h-4 w-4 stroke-[1.5]" />
            Register
          </Button>
        </motion.div>
      </DrawerTrigger>
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle>Register New Face</DrawerTitle>
          <DrawerDescription>Capture and register a new face for the system</DrawerDescription>
        </DrawerHeader>
        <motion.div
          className="space-y-4 px-4 pb-6"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <div className="space-y-2">
            <Label htmlFor="name" className="text-sm font-medium">
              Full Name
            </Label>
            <Input
              id="name"
              placeholder="Enter full name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded-lg border-border"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">Face Capture</Label>
            <div className="flex h-32 items-center justify-center rounded-xl border border-dashed border-border bg-muted">
              <span className="text-sm text-muted-foreground font-medium">Capture area</span>
            </div>
          </div>
          <div className="flex gap-3">
            <DrawerClose asChild>
              <Button variant="outline" className="flex-1 rounded-lg bg-transparent">
                Cancel
              </Button>
            </DrawerClose>
            <motion.div className="flex-1" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button onClick={handleSubmit} disabled={!name || isSubmitting} className="w-full rounded-lg">
                {isSubmitting ? "Registering..." : "Register Face"}
              </Button>
            </motion.div>
          </div>
        </motion.div>
      </DrawerContent>
    </Drawer>
  )
}
