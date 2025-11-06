"use client"

import { useState, useRef } from "react"
import { UserPlus, X, Upload } from "lucide-react"
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
import { motion, AnimatePresence } from "framer-motion"
import { registerFace } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Badge } from "@/components/ui/badge"

const MAX_IMAGES = 5

export function RegisterDrawer() {
  const [name, setName] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [selectedImages, setSelectedImages] = useState<File[]>([])
  const [previewUrls, setPreviewUrls] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    const remainingSlots = MAX_IMAGES - selectedImages.length
    const filesToAdd = files.slice(0, remainingSlots)
    
    if (files.length > remainingSlots) {
      toast({
        title: "Too many images",
        description: `Maximum ${MAX_IMAGES} images allowed. Only ${remainingSlots} added.`,
        variant: "destructive",
      })
    }

    const newFiles = [...selectedImages, ...filesToAdd]
    const newUrls = [...previewUrls, ...filesToAdd.map(file => URL.createObjectURL(file))]

    setSelectedImages(newFiles)
    setPreviewUrls(newUrls)
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const removeImage = (index: number) => {
    const newFiles = selectedImages.filter((_, i) => i !== index)
    const urlToRevoke = previewUrls[index]
    URL.revokeObjectURL(urlToRevoke)
    const newUrls = previewUrls.filter((_, i) => i !== index)
    
    setSelectedImages(newFiles)
    setPreviewUrls(newUrls)
  }

  const handleSubmit = async () => {
    if (!name.trim()) {
      toast({
        title: "Missing name",
        description: "Please enter a name",
        variant: "destructive",
      })
      return
    }

    if (selectedImages.length === 0) {
      toast({
        title: "No images selected",
        description: `Please select at least 1 image (up to ${MAX_IMAGES})`,
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)
    try {
      // Register all images for the same user
      for (let i = 0; i < selectedImages.length; i++) {
        await registerFace(name.trim(), selectedImages[i], "A")
      }
      
      toast({
        title: "Registration successful",
        description: `Registered ${selectedImages.length} image${selectedImages.length > 1 ? 's' : ''} for ${name}`,
      })
      
      // Trigger registry update event to refresh registry info in other components
      window.dispatchEvent(new Event('registry-updated'))
      
      // Clear form
      setName("")
      previewUrls.forEach(url => URL.revokeObjectURL(url))
      setSelectedImages([])
      setPreviewUrls([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
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

  return (
    <Drawer>
      <DrawerTrigger asChild>
        <motion.div className="w-full" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button variant="outline" className="w-full gap-2 rounded-lg h-10 bg-transparent">
            <UserPlus className="h-4 w-4 stroke-[1.5]" />
            Register
          </Button>
        </motion.div>
      </DrawerTrigger>
      <DrawerContent className="max-h-[90vh]">
        <DrawerHeader className="sticky top-0 bg-background z-10 border-b">
          <DrawerTitle>Register New Face</DrawerTitle>
          <DrawerDescription>
            Upload up to {MAX_IMAGES} images for better verification accuracy
          </DrawerDescription>
        </DrawerHeader>
        <div className="overflow-y-auto">
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
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Face Images</Label>
                <Badge variant="outline" className="text-xs">
                  {selectedImages.length} / {MAX_IMAGES}
                </Badge>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageSelect}
                className="hidden"
                disabled={selectedImages.length >= MAX_IMAGES}
              />
              
              {selectedImages.length > 0 ? (
                <div className="grid grid-cols-2 gap-3">
                  <AnimatePresence>
                    {previewUrls.map((url, index) => (
                      <motion.div
                        key={index}
                        className="relative group aspect-square rounded-xl border-2 border-border bg-muted overflow-hidden"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ duration: 0.2 }}
                      >
                        <img
                          src={url}
                          alt={`Preview ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                        <button
                          onClick={() => removeImage(index)}
                          className="absolute top-1 right-1 p-1 rounded-full bg-red-500/90 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                        <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] px-2 py-1 text-center">
                          Image {index + 1}
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  {selectedImages.length < MAX_IMAGES && (
                    <motion.div
                      className="aspect-square rounded-xl border-2 border-dashed border-border bg-muted/50 cursor-pointer hover:bg-muted/80 transition-colors flex flex-col items-center justify-center gap-2"
                      onClick={() => fileInputRef.current?.click()}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Upload className="h-8 w-8 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground font-medium text-center px-2">
                        Add Image
                      </span>
                    </motion.div>
                  )}
                </div>
              ) : (
                <div 
                  className="flex h-32 items-center justify-center rounded-xl border-2 border-dashed border-border bg-muted cursor-pointer hover:bg-muted/80 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="text-center">
                    <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                    <span className="text-sm text-muted-foreground font-medium">Click to select images</span>
                    <span className="text-xs text-muted-foreground block mt-1">Up to {MAX_IMAGES} images</span>
                  </div>
                </div>
              )}
              {selectedImages.length < MAX_IMAGES && selectedImages.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Add More Images ({MAX_IMAGES - selectedImages.length} remaining)
                </Button>
              )}
            </div>
            
            <div className="flex gap-3 pt-2">
            <DrawerClose asChild>
              <Button variant="outline" className="flex-1 rounded-lg bg-transparent">
                Cancel
              </Button>
            </DrawerClose>
            <motion.div className="flex-1" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  onClick={handleSubmit} 
                  disabled={!name || selectedImages.length === 0 || isSubmitting} 
                  className="w-full rounded-lg"
                >
                  {isSubmitting 
                    ? `Registering... (${selectedImages.length} images)` 
                    : `Register ${selectedImages.length || ''} Image${selectedImages.length !== 1 ? 's' : ''}`
                  }
              </Button>
            </motion.div>
          </div>
        </motion.div>
        </div>
      </DrawerContent>
    </Drawer>
  )
}
