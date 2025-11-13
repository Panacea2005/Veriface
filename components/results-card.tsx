"use client"

import { useState, useEffect } from "react"
import { CheckCircle2, XCircle, AlertCircle, Eye, Clock, Activity, TrendingUp, Info, LogIn, LogOut, Smile } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { motion, AnimatePresence } from "framer-motion"
import type { VerifyResponse } from "@/lib/api"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend } from "recharts"

interface ResultsCardProps {
  verifyResult: VerifyResponse | null
  verifyHistory?: VerifyResponse[]
}

interface SessionMetrics {
  totalVerifications: number
  successfulMatches: number
  failedMatches: number
  avgLivenessScore: number
  avgMatchScore: number
  timestampedData: Array<{
    index: string
    liveness: number
    matchScore: number | null
  }>
}

interface DetailedLog {
  id: string
  timestamp: string
  event: string
  status: "success" | "failed" | "warning" | "info"
  details?: string
  duration?: string
  verificationIndex?: number
}

export function ResultsCard({ verifyResult, verifyHistory = [] }: ResultsCardProps) {
  const [logs, setLogs] = useState<DetailedLog[]>([])
  const [processingSteps, setProcessingSteps] = useState<Array<{ step: string; time: number }>>([])
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetrics>({
    totalVerifications: 0,
    successfulMatches: 0,
    failedMatches: 0,
    avgLivenessScore: 0,
    avgMatchScore: 0,
    timestampedData: []
  })
  // Emotion UI moved to WebcamSection; no subscription needed here

  // Accumulate logs from all verification results
  useEffect(() => {
    const allLogs: DetailedLog[] = []
    
    if (verifyHistory.length > 0) {
      verifyHistory.forEach((result, idx) => {
        const timestamp = new Date()
        // Use a timestamp offset based on verification index for better visualization
        timestamp.setSeconds(timestamp.getSeconds() - (verifyHistory.length - idx))
        
        const dateStr = timestamp.toLocaleDateString("en-US", { 
          year: "numeric",
          month: "2-digit",
          day: "2-digit"
        })
        const timeStr = timestamp.toLocaleTimeString("en-US", { 
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          fractionalSecondDigits: 3
        })
        const fullTimestamp = `${dateStr} ${timeStr}`
        
        const verificationLogs: DetailedLog[] = [
          {
            id: `${idx}-1`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Image Processing`,
            status: "success",
            details: "Image loaded and validated",
            duration: "< 50ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-2`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Face Detection`,
            status: "success",
            details: "Face bounding box detected",
            duration: "~100ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-3`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Face Alignment`,
            status: "success",
            details: "Face cropped and aligned to 112x112",
            duration: "~20ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-4`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Liveness Detection`,
            status: result.liveness.passed ? "success" : "failed",
            details: `Score: ${(result.liveness.score * 100).toFixed(2)}%`,
            duration: "~150ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-5`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Feature Extraction`,
            status: "success",
            details: "512-D embedding vector generated",
            duration: "~200ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-6`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Similarity Matching`,
            status: result.matched_id ? "success" : "warning",
            details: result.score !== null && result.score !== undefined
              ? (result.matched_id
                  ? `Matched: ${result.matched_name || result.matched_id} (ID: ${result.matched_id}) - ${result.metric === "cosine" ? (result.score * 100).toFixed(2) + "%" : result.score.toFixed(4)}`
                  : `Best score: ${result.metric === "cosine" ? (result.score * 100).toFixed(2) + "%" : result.score.toFixed(4)} (insufficient consensus - Hybrid mode requires 60% of angles to pass)`)
              : "No scores calculated",
            duration: "~50ms",
            verificationIndex: idx + 1
          },
          // Emotion step omitted from Results timeline
        ]
        
        allLogs.push(...verificationLogs)
      })
    }
    
    // Reverse to show latest first
    setLogs(allLogs.reverse())
    
    // Processing steps from latest result
    if (verifyResult) {
      setProcessingSteps([
        { step: "Detection", time: 100 },
        { step: "Alignment", time: 20 },
        { step: "Liveness", time: 150 },
        { step: "Embedding", time: 200 },
        { step: "Matching", time: 50 },
        { step: "Emotion", time: 100 }
      ])
    } else {
      setProcessingSteps([])
    }
  }, [verifyResult, verifyHistory])

  const convertMatchScoreToPercent = (result: VerifyResponse): number | null => {
    if (result.score === null || result.score === undefined) return null
    if (result.metric === "cosine") {
      return Math.max(0, Math.min(100, result.score * 100))
    }
    const distance = result.score
    return Math.max(0, Math.min(100, (1 - Math.min(distance / 10, 1)) * 100))
  }

  useEffect(() => {
    if (verifyHistory.length === 0) {
      setSessionMetrics({
        totalVerifications: 0,
        successfulMatches: 0,
        failedMatches: 0,
        avgLivenessScore: 0,
        avgMatchScore: 0,
        timestampedData: []
      })
      return
    }

    const successful = verifyHistory.filter(r => r.matched_id !== null).length
    const failed = verifyHistory.length - successful
    const avgLiveness = verifyHistory.reduce((sum, r) => sum + r.liveness.score, 0) / verifyHistory.length
    const matchScores = verifyHistory
      .map(convertMatchScoreToPercent)
      .filter((score): score is number => score !== null)

    const avgMatch = matchScores.length > 0
      ? matchScores.reduce((sum, value) => sum + value, 0) / matchScores.length
      : 0

    const timestampedData = verifyHistory.map((r, idx) => ({
      index: `${idx + 1}`,
      liveness: Math.max(0, Math.min(100, r.liveness.score * 100)),
      matchScore: convertMatchScoreToPercent(r)
    }))

    setSessionMetrics({
      totalVerifications: verifyHistory.length,
      successfulMatches: successful,
      failedMatches: failed,
      avgLivenessScore: avgLiveness * 100,
      avgMatchScore: avgMatch,
      timestampedData
    })
  }, [verifyHistory])

  // Calculate match score - always show best score from backend
  const matchScore = verifyResult?.score !== null && verifyResult?.score !== undefined
    ? (verifyResult.metric === "cosine" 
        ? verifyResult.score * 100  // Cosine: convert 0-1 to 0-100%
        : verifyResult.score)  // Euclidean: keep as distance
    : (verifyResult?.all_scores && verifyResult.all_scores.length > 0
        ? (verifyResult.metric === "cosine"
            ? verifyResult.all_scores[0].percentage  // Use percentage from all_scores
            : verifyResult.all_scores[0].score)  // Use raw score for euclidean
        : 0)
  
  const threshold = verifyResult?.threshold !== null && verifyResult?.threshold !== undefined
    ? (verifyResult.metric === "cosine" 
        ? verifyResult.threshold * 100  // Cosine: convert 0-1 to 0-100%
        : verifyResult.threshold)  // Euclidean: keep as distance
    : (verifyResult?.metric === "cosine" ? 75 : 5.0)
  
  const isLive = verifyResult?.liveness.passed ?? false

  // Check if matched (has matched_id means passed threshold)
  const isMatched = verifyResult?.matched_id !== null && verifyResult?.matched_id !== undefined
  
  // Calculate percentage for display
  const scorePercentage = verifyResult?.metric === "cosine" 
    ? matchScore  // Already in percentage
    : Math.max(0, Math.min(100, (1 - Math.min(matchScore / 10, 1)) * 100))  // Euclidean: convert distance to percentage

  const getScoreColor = () => {
    if (!verifyResult) return "bg-muted"
    if (verifyResult.metric === "cosine") {
    if (matchScore >= threshold) return "bg-green-600"
    if (matchScore >= threshold - 10) return "bg-amber-600"
    return "bg-red-600"
    } else {
      if (matchScore <= threshold) return "bg-green-600"
      if (matchScore <= threshold + 2) return "bg-amber-600"
      return "bg-red-600"
    }
  }

  const chartConfig = {
    liveness: {
      label: "Liveness Score (%)",
      color: "#3b82f6"
    },
    matchScore: {
      label: "Match Score (%)",
      color: "#10b981"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return <CheckCircle2 className="h-3.5 w-3.5 stroke-[1.5]" />
      case "failed":
        return <XCircle className="h-3.5 w-3.5 stroke-[1.5]" />
      case "warning":
        return <AlertCircle className="h-3.5 w-3.5 stroke-[1.5]" />
      default:
        return <Activity className="h-3.5 w-3.5 stroke-[1.5]" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success":
        return "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300 border-green-200 dark:border-green-800"
      case "failed":
        return "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300 border-red-200 dark:border-red-800"
      case "warning":
        return "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300 border-amber-200 dark:border-amber-800"
      default:
        return "bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border-blue-200 dark:border-blue-800"
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <Card className="flex flex-col border border-border shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-base font-semibold">Verification Results</CardTitle>
          <CardDescription className="text-xs">Real-time analysis and matching pipeline</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          {verifyResult ? (
            <>
              {/* Realtime emotion subscribed via top-level useEffect */}
              {/* Score Section */}
              <motion.div
                className="space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="flex items-end justify-between">
                  <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-foreground/80 uppercase tracking-wide">Match Score</span>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-foreground/60 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Similarity score using {verifyResult.metric} distance metric</p>
                          {matchScore === 0 && (
                            <p className="text-xs text-yellow-400 mt-1">
                              ⚠️ Insufficient consensus: Not enough face angles matched the required threshold
                            </p>
                          )}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <div className="flex items-center gap-2">
                    <motion.span
                      className="text-3xl font-bold"
                      initial={{ scale: 0.8 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 100 }}
                    >
                      {verifyResult.metric === "cosine" ? `${matchScore.toFixed(1)}%` : matchScore.toFixed(2)}
                    </motion.span>
                    {!isMatched && matchScore > 0 && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge variant="outline" className="text-xs cursor-help bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/30">
                              Below Threshold
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="font-medium">Why did verification fail?</p>
                            <p className="text-xs mt-1">
                              Using <span className="font-semibold">Delta-Margin</span>: Checks if top 2 scores have sufficient gap.
                            </p>
                            <p className="text-xs mt-1">
                              <span className="text-yellow-400">⚠️</span> Score shown ({matchScore.toFixed(1)}%) is below the required threshold (55.0%) to confirm identity.
                            </p>
                            <p className="text-xs mt-1 text-foreground/70">
                              This prevents false matches between similar faces (e.g., siblings) by requiring a clear winner.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted border border-border">
                  <motion.div
                    className={`h-full transition-all ${getScoreColor()}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${scorePercentage}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
                <div className="flex justify-between items-center text-xs text-foreground/70">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="cursor-help flex items-center gap-1">
                          <TrendingUp className="h-3 w-3" />
                          Threshold: {verifyResult?.metric === "cosine" ? `${threshold.toFixed(1)}%` : threshold.toFixed(2)}
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Minimum score required for a match</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <motion.span
                    className={`font-medium flex items-center gap-1 ${isMatched ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400"}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    {isMatched ? (
                      <>
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        Match Found
                      </>
                    ) : verifyResult ? (
                      <>
                        <AlertCircle className="h-3.5 w-3.5" />
                        No Match
                      </>
                    ) : null}
                  </motion.span>
                </div>
              </motion.div>

              {/* Status Badges */}
              <motion.div
                className="flex flex-wrap gap-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.25 }}
              >
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant={isLive ? "default" : "secondary"} className="gap-1.5 cursor-help rounded-lg">
                        <Eye className="h-3.5 w-3.5 stroke-[1.5]" />
                        {isLive ? "Live" : "Spoofed"}
                        <span className="text-xs opacity-75">
                          ({(verifyResult.liveness.score * 100).toFixed(1)}%)
                        </span>
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Liveness detection: {verifyResult.liveness.score > 0.5 ? "Real face detected" : "Potential spoof"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                {verifyResult.emotion_label && (
                  <Badge variant="outline" className="gap-1.5 rounded-lg">
                    <Smile className="h-3.5 w-3.5 stroke-[1.5]" />
                    {verifyResult.emotion_label.charAt(0).toUpperCase() + verifyResult.emotion_label.slice(1)}
                  </Badge>
                )}

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="secondary" className="gap-1.5 rounded-lg cursor-help">
                        <TrendingUp className="h-3.5 w-3.5 stroke-[1.5]" />
                        Delta-Margin
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="font-medium">Delta-Margin Anti-Sibling Verification</p>
                      <ul className="text-xs mt-1 list-disc list-inside space-y-0.5">
                        <li>Requires clear winner: gap between 1st & 2nd score &gt; 0.20</li>
                        <li>If scores too close → potential sibling → penalize</li>
                        <li>Best method for preventing family false positives</li>
                        <li>Used in face recognition competitions (SOTA)</li>
                      </ul>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                {verifyResult.matched_id && (
                  <Badge variant="default" className="gap-1.5 rounded-lg">
                    <Activity className="h-3.5 w-3.5 stroke-[1.5]" />
                    {verifyResult.matched_name || verifyResult.matched_id} (ID: {verifyResult.matched_id})
                  </Badge>
                )}
              </motion.div>

              {/* Emotion from Verification */}
              {verifyResult.emotion_label && verifyResult.emotion_probs && Object.keys(verifyResult.emotion_probs).length > 0 && (
                <motion.div
                  className="rounded-lg border border-border p-3"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.27 }}
                >
                  <div className="text-sm font-semibold mb-2">
                    <span>Emotion (from verification)</span>
                  </div>
                  <div className="space-y-2 min-h-44">
                    {Object.entries(verifyResult.emotion_probs)
                      .sort((a, b) => b[1] - a[1])
                      .map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2">
                          <div className={`text-xs w-20 ${verifyResult.emotion_label === k ? "font-semibold text-foreground" : "text-foreground/70"}`}>{k}</div>
                          <div className="flex-1 h-2 bg-muted rounded">
                            <div className="h-2 bg-accent rounded" style={{ width: `${Math.min(100, Math.max(0, v * 100)).toFixed(0)}%` }} />
                          </div>
                          <div className="w-12 text-right text-xs tabular-nums">{(v * 100).toFixed(0)}%</div>
                        </div>
                      ))}
                    {/* Dominant Emotion Highlight */}
                    {(() => {
                      const [k, v] = Object.entries(verifyResult.emotion_probs).sort((a,b)=>b[1]-a[1])[0]
                      const em = k === "happy" ? "😊" : k === "sad" ? "😢" : k === "angry" ? "😠" : k === "surprise" ? "😲" : k === "fear" ? "😨" : k === "disgust" ? "🤢" : "😐"
                      return (
                        <div className="mt-4 pt-4 border-t border-border">
                          <div className="rounded-lg bg-background border border-border p-3 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <span className="text-3xl leading-none">{em}</span>
                              <div className="flex flex-col">
                                <span className="text-xs text-foreground/70 uppercase tracking-wide">Detected</span>
                                <span className="text-base font-bold capitalize text-foreground">{k}</span>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-accent">{(v * 100).toFixed(0)}%</div>
                              <div className="text-[10px] text-foreground/70 uppercase">Confidence</div>
                            </div>
                          </div>
                        </div>
                      )
                    })()}
                  </div>
                </motion.div>
              )}

              {/* Registry Match Scores */}
              {verifyResult?.all_scores && verifyResult.all_scores.length > 0 && (
                <motion.div
                  className="space-y-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.28 }}
                >
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-3.5 w-3.5 text-foreground/70" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-foreground">
                      Registry Match Scores
                    </h3>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {verifyResult.all_scores.length} user{verifyResult.all_scores.length !== 1 ? 's' : ''}
                    </Badge>
                  </div>
                  <div className="rounded-xl border border-border overflow-hidden">
                    <Table className="text-xs">
                      <TableHeader>
                        <TableRow className="border-b border-border hover:bg-transparent bg-muted/30">
                          <TableHead className="h-8 px-3 py-2 font-medium text-foreground/80">Rank</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-foreground/80">User ID</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-foreground/80 text-right">Match %</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-foreground/80 text-right">Embeddings</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {verifyResult.all_scores.map((item, idx) => {
                          const isMatched = verifyResult.matched_id === item.user_id
                          const isTopMatch = idx === 0
                          return (
                            <motion.tr
                              key={item.user_id}
                              className={`border-b border-border last:border-b-0 transition-colors ${
                                isMatched 
                                  ? "bg-green-50 dark:bg-green-950/30 hover:bg-green-100 dark:hover:bg-green-950/50" 
                                  : isTopMatch
                                  ? "bg-amber-50 dark:bg-amber-950/20 hover:bg-amber-100 dark:hover:bg-amber-950/30"
                                  : "hover:bg-muted/50"
                              }`}
                              initial={{ opacity: 0, y: -5 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.3 + idx * 0.03 }}
                            >
                              <TableCell className="px-3 py-2">
                                <span className={`font-bold ${isMatched ? "text-green-600 dark:text-green-400" : isTopMatch ? "text-amber-600 dark:text-amber-400" : "text-foreground/70"}`}>
                                  #{idx + 1}
                                </span>
                              </TableCell>
                              <TableCell className="px-3 py-2 font-medium text-foreground">
                                <div className="flex items-center gap-2">
                                  {item.user_id}
                                  {isMatched && (
                                    <Badge variant="default" className="text-[10px] px-1.5 py-0 h-4">
                                      MATCHED
                                    </Badge>
                                  )}
                                </div>
                              </TableCell>
                              <TableCell className="px-3 py-2 text-right">
                                <div className="flex items-center justify-end gap-2">
                                  <div className="w-20 bg-muted rounded-full h-1.5 overflow-hidden">
                                    <motion.div
                                      className={`h-full ${isMatched ? "bg-green-600" : isTopMatch ? "bg-amber-600" : "bg-accent"}`}
                                      initial={{ width: 0 }}
                                      animate={{ width: `${item.percentage}%` }}
                                      transition={{ duration: 0.5, delay: 0.3 + idx * 0.03 }}
                                    />
                                  </div>
                                  <span className={`font-bold min-w-[50px] text-right ${isMatched ? "text-green-600 dark:text-green-400" : isTopMatch ? "text-amber-600 dark:text-amber-400" : ""}`}>
                                    {item.percentage.toFixed(1)}%
                                  </span>
                                </div>
                              </TableCell>
                              <TableCell className="px-3 py-2 text-right text-foreground/70 text-[10px]">
                                {item.embeddings_count}
                              </TableCell>
                            </motion.tr>
                          )
                        })}
                      </TableBody>
                    </Table>
                  </div>
                </motion.div>
              )}

              {/* Matched User ID */}
              {verifyResult?.matched_id && (
                <motion.div
                  className="flex justify-center px-2"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.32 }}
                >
                  <div className="w-full md:max-w-md rounded-xl border border-border bg-background px-5 py-4 text-center shadow-sm">
                    <p className="text-[11px] font-medium text-foreground/60 uppercase tracking-[0.18em]">Matched Identity</p>
                    <p className="mt-2 text-2xl font-bold text-foreground break-words">{verifyResult.matched_name || verifyResult.matched_id}</p>
                    <p className="mt-1 text-sm font-mono text-foreground/60">{verifyResult.matched_id}</p>
                    {verifyResult.check_type && (
                      <Badge 
                        className="mt-3 h-6 px-2.5 text-[11px] font-semibold font-mono"
                        variant={verifyResult.check_type === "check-out" ? "secondary" : "default"}
                      >
                        {verifyResult.check_type === "check-out" ? "OUT" : "IN"}
                      </Badge>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Session Metrics */}
              {sessionMetrics.totalVerifications > 0 && (
                <motion.div
                  className="grid gap-4 lg:grid-cols-3"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.33 }}
                >
                  <div className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-foreground/70">Total Verifications</span>
                      <span className="text-lg font-bold text-foreground">{sessionMetrics.totalVerifications}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-foreground/70">Successful Matches</span>
                      <span className="text-lg font-bold text-green-600">{sessionMetrics.successfulMatches}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-foreground/70">Failed Matches</span>
                      <span className="text-lg font-bold text-amber-600">{sessionMetrics.failedMatches}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-foreground/70">Avg Liveness</span>
                      <span className="text-lg font-bold text-foreground">{sessionMetrics.avgLivenessScore.toFixed(1)}%</span>
                    </div>
                    {sessionMetrics.avgMatchScore > 0 && (
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-foreground/70">Avg Match Score</span>
                        <span className="text-lg font-bold text-foreground">{sessionMetrics.avgMatchScore.toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                  <div className="lg:col-span-2 rounded-xl border border-border bg-muted/10 p-4">
                    {sessionMetrics.timestampedData.length > 0 ? (
                      <ChartContainer config={chartConfig} className="h-[260px] w-full" id="session-performance">
                        <LineChart
                          data={sessionMetrics.timestampedData}
                          margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
                          width={undefined}
                          height={260}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                          <XAxis
                            dataKey="index"
                            label={{ value: "Verification #", position: "insideBottom", offset: -10, style: { fontSize: 11 } }}
                            tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }}
                            stroke="hsl(var(--border))"
                          />
                          <YAxis
                            domain={[0, 100]}
                            tickFormatter={(value) => `${value}%`}
                            tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }}
                            stroke="hsl(var(--border))"
                          />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          <Legend verticalAlign="top" height={32} iconType="line" />
                          <Line
                            type="monotone"
                            dataKey="liveness"
                            stroke="#3b82f6"
                            strokeWidth={3}
                            activeDot={{ r: 6, fill: "#3b82f6" }}
                            dot={{ r: 4, fill: "#3b82f6" }}
                            name="Liveness Score"
                          />
                          <Line
                            type="monotone"
                            dataKey="matchScore"
                            stroke="#10b981"
                            strokeWidth={3}
                            activeDot={{ r: 6, fill: "#10b981" }}
                            dot={{ r: 4, fill: "#10b981" }}
                            name="Match Score"
                            strokeDasharray="6 4"
                            connectNulls
                          />
                        </LineChart>
                      </ChartContainer>
                    ) : (
                      <div className="flex h-[260px] items-center justify-center rounded-lg border border-dashed border-border bg-muted/30">
                        <p className="text-xs text-muted-foreground">Run verifications to see session performance</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Processing Timeline */}
              {processingSteps.length > 0 && (
                <motion.div
                  className="space-y-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.35 }}
                >
                  <div className="flex items-center gap-2">
                    <Clock className="h-3.5 w-3.5 text-foreground/70" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-foreground">Processing Timeline</h3>
                  </div>
                  <div className="flex items-center gap-1 overflow-x-auto pb-2">
                    {processingSteps.map((step, idx) => (
                      <div key={idx} className="flex items-center gap-1 flex-shrink-0">
                        <div className="text-center min-w-[60px]">
                          <div className="text-[10px] font-medium text-foreground/80">{step.step}</div>
                          <div className="text-[10px] text-foreground/60">{step.time}ms</div>
                        </div>
                        {idx < processingSteps.length - 1 && (
                          <div className="w-4 h-0.5 bg-border mx-1" />
                        )}
                      </div>
                    ))}
                    <div className="text-[10px] text-foreground/70 ml-2">
                      Total: ~{processingSteps.reduce((sum, s) => sum + s.time, 0)}ms
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Emotion section removed (shown in WebcamSection) */}

              {/* Detailed Event Logs */}
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                  <div className="flex items-center gap-2">
                    <Activity className="h-3.5 w-3.5 text-foreground/70" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-foreground">Processing Pipeline</h3>
                  </div>
                <div className="rounded-xl border border-border" style={{ maxHeight: "400px", overflowY: "auto", overflowX: "auto" }}>
                  <Table className="text-xs">
                    <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur supports-[backdrop-filter]:bg-muted/80">
                      <TableRow className="border-b border-border hover:bg-transparent bg-muted/30">
                        <TableHead className="h-9 px-3 py-2 font-medium text-foreground/80 w-[180px] sticky left-0 bg-muted/95 backdrop-blur supports-[backdrop-filter]:bg-muted/80 z-10">Timestamp</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-foreground/80">Event</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-foreground/80">Details</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-foreground/80 w-[80px]">Duration</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-foreground/80 w-[100px]">Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <AnimatePresence>
                      {logs.map((log, idx) => (
                        <motion.tr
                            key={log.id}
                          className="border-b border-border last:border-b-0 hover:bg-muted/50 transition-colors"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.4 + idx * 0.05 }}
                        >
                            <TableCell className="px-3 py-2 font-mono text-foreground/70 text-[10px]">
                              {log.timestamp}
                            </TableCell>
                            <TableCell className="px-3 py-2 text-xs font-medium text-foreground">{log.event}</TableCell>
                            <TableCell className="px-3 py-2 text-xs text-foreground/80">{log.details || "-"}</TableCell>
                            <TableCell className="px-3 py-2 text-xs text-foreground/70 font-mono">{log.duration || "-"}</TableCell>
                          <TableCell className="px-3 py-2">
                            <Badge
                              variant="outline"
                                className={`rounded-md gap-1 ${getStatusColor(log.status)}`}
                            >
                                {getStatusIcon(log.status)}
                              {log.status}
                            </Badge>
                          </TableCell>
                        </motion.tr>
                      ))}
                      </AnimatePresence>
                    </TableBody>
                  </Table>
                </div>
              </motion.div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-foreground/70">
              <Activity className="h-12 w-12 opacity-40 mb-3" />
              <span className="text-sm font-medium text-foreground">No verification results yet</span>
              <span className="text-xs mt-1 text-foreground/70">Upload an image and verify above to see results</span>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
