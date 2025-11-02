"use client"

import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { WebcamSection } from "@/components/webcam-section"
import { ResultsCard } from "@/components/results-card"
import { EvaluationSection } from "@/components/evaluation-section"
import { Footer } from "@/components/footer"
import { Toaster } from "@/components/ui/toaster"
import type { VerifyResponse } from "@/lib/api"

export default function Page() {
  const [isDark, setIsDark] = useState(false)
  const [isOnline, setIsOnline] = useState(true)
  const [mounted, setMounted] = useState(false)
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null)
  const [verifyHistory, setVerifyHistory] = useState<VerifyResponse[]>([])

  useEffect(() => {
    setMounted(true)
    const savedTheme = localStorage.getItem("theme")
    if (savedTheme === "dark") {
      setIsDark(true)
      document.documentElement.classList.add("dark")
    } else if (savedTheme === "light") {
      setIsDark(false)
      document.documentElement.classList.remove("dark")
    }
  }, [])

  const toggleTheme = () => {
    const newIsDark = !isDark
    setIsDark(newIsDark)
    localStorage.setItem("theme", newIsDark ? "dark" : "light")
    if (newIsDark) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
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
    <div className={isDark ? "dark" : ""}>
      <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200">
        {/* Header */}
        <Header isDark={isDark} onToggleTheme={toggleTheme} />

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
          </div>
        </main>

        {/* Evaluation Section - At the bottom, always visible */}
        <section id="evaluation" className="py-16 bg-muted/30 border-t border-border">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <EvaluationSection verifyResults={verifyHistory} />
          </div>
        </section>

        {/* Footer */}
        <Footer />
        
        {/* Toast notifications */}
        <Toaster />
      </div>
    </div>
  )
}
