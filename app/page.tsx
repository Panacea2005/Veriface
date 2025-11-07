"use client"

import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { WebcamSection } from "@/components/webcam-section"
import { ResultsCard } from "@/components/results-card"
import { AttendanceHistory } from "@/components/attendance-history"
import { Footer } from "@/components/footer"
import { Toaster } from "@/components/ui/toaster"
import type { VerifyResponse } from "@/lib/api"

export default function Page() {
  const [isDark, setIsDark] = useState(false)
  const [isBaseTheme, setIsBaseTheme] = useState(false)
  const [isOnline, setIsOnline] = useState(true)
  const [mounted, setMounted] = useState(false)
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null)
  const [verifyHistory, setVerifyHistory] = useState<VerifyResponse[]>([])

  useEffect(() => {
    setMounted(true)
    const root = document.documentElement
    const savedTheme = localStorage.getItem("theme")
    const savedStyle = localStorage.getItem("theme-style")

    if (savedTheme === "dark") {
      setIsDark(true)
      root.classList.add("dark")
    } else if (savedTheme === "light") {
      setIsDark(false)
      root.classList.remove("dark")
    }

    if (savedStyle === "base") {
      setIsBaseTheme(true)
      root.classList.add("theme-base")
    } else {
      root.classList.remove("theme-base")
    }
  }, [])

  const toggleTheme = () => {
    const newIsDark = !isDark
    setIsDark(newIsDark)
    localStorage.setItem("theme", newIsDark ? "dark" : "light")
    const root = document.documentElement
    if (newIsDark) {
      root.classList.add("dark")
    } else {
      root.classList.remove("dark")
    }
  }

  const toggleThemeStyle = () => {
    const next = !isBaseTheme
    setIsBaseTheme(next)
    localStorage.setItem("theme-style", next ? "base" : "neo")
    const root = document.documentElement
    if (next) {
      root.classList.add("theme-base")
    } else {
      root.classList.remove("theme-base")
    }
  }

  const handleVerifyResult = (result: VerifyResponse | null) => {
    setVerifyResult(result)
    if (result) {
      setVerifyHistory(prev => [...prev, result].slice(-50)) // Keep last 50 results
    }
  }

  if (!mounted) return null

  return (
    <div className={`${isDark ? "dark" : ""} ${isBaseTheme ? "theme-base" : ""}`}>
      <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200">
        {/* Header */}
        <Header
          isDark={isDark}
          onToggleTheme={toggleTheme}
          isBaseTheme={isBaseTheme}
          onToggleThemeStyle={toggleThemeStyle}
        />

        {/* Main Content */}
        <main className="flex-1 py-16">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 space-y-8">
            {/* Webcam Section - Full Width */}
            <div id="webcam">
              <WebcamSection onVerifyResult={handleVerifyResult} />
            </div>

            {/* Results Section - Full Width */}
            <div id="results">
              <ResultsCard verifyResult={verifyResult} verifyHistory={verifyHistory} />
            </div>

            {/* Attendance History Section - Full Width */}
            <div id="attendance">
              <AttendanceHistory verifyResult={verifyResult} />
            </div>
          </div>
        </main>

        {/* Footer */}
        <Footer />
        
        {/* Toast notifications */}
        <Toaster />
      </div>
    </div>
  )
}
