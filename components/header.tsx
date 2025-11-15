"use client"

import { Moon, Sun, Palette } from "lucide-react"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import Image from "next/image"

interface HeaderProps {
  isDark: boolean
  onToggleTheme: () => void
  isBaseTheme: boolean
  onToggleThemeStyle: () => void
}

export function Header({ isDark, onToggleTheme, isBaseTheme, onToggleThemeStyle }: HeaderProps) {
  return (
    <motion.header
      className="py-4 sm:py-6 bg-background"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 w-full">
        <div className="flex items-center justify-between gap-6 w-full min-h-[80px]">
          {/* Logo and Brand */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <Image src="/logo.png" alt="Veriface Logo" width={96} height={96} priority className="w-20 h-20 sm:w-24 sm:h-24" />
            <div className="flex flex-col items-start">
              <span className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-extrabold tracking-tight leading-none uppercase">
                Veriface
              </span>
              <span className="font-medium text-sm sm:text-base lg:text-lg text-muted-foreground mt-1">
                Face Recognition Attendance System
              </span>
            </div>
          </div>
          {/* Theme Toggles */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button
                variant="outline"
                size="sm"
                onClick={onToggleThemeStyle}
                className={`rounded-lg px-3 h-9 gap-2 font-semibold transition-colors ${isBaseTheme ? "bg-accent text-accent-foreground border-accent/40" : "hover:bg-muted"}`}
                aria-label="Toggle neo brutalism / claude theme"
              >
                <Palette className="h-4 w-4" />
                {isBaseTheme ? "Claude" : "Neo"}
              </Button>
            </motion.div>
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button variant="outline" size="icon" onClick={onToggleTheme} className="h-10 w-10 rounded-lg" aria-label="Toggle dark mode">
              {isDark ? <Sun className="h-5 w-5 stroke-[1.5]" /> : <Moon className="h-5 w-5 stroke-[1.5]" />}
                <span className="sr-only">Toggle dark mode</span>
            </Button>
          </motion.div>
          </div>
        </div>
      </div>
    </motion.header>
  );
}
