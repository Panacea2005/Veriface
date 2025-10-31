"use client"

import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { WebcamSection } from "@/components/webcam-section"
import { ResultsCard } from "@/components/results-card"
import { EvaluationSection } from "@/components/evaluation-section"
import { Footer } from "@/components/footer"

export default function Page() {
  const [isDark, setIsDark] = useState(false)
  const [isOnline, setIsOnline] = useState(true)
  const [mounted, setMounted] = useState(false)

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
              <WebcamSection />
            </div>

            {/* Results Section - Full Width */}
            <div id="results">
              <ResultsCard />
            </div>

            {/* Evaluation Section - Full Width */}
            <div id="evaluation">
              <EvaluationSection />
            </div>
          </div>
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </div>
  )
}
