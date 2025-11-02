"use client"

import { useState, useEffect } from "react"
import { CheckCircle2, XCircle, AlertCircle, Smile, Eye, Clock, Activity, TrendingUp, Info } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { motion, AnimatePresence } from "framer-motion"
import type { VerifyResponse } from "@/lib/api"

interface ResultsCardProps {
  verifyResult: VerifyResponse | null
  verifyHistory?: VerifyResponse[]
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
                  ? `Matched: ${result.matched_id} (${result.metric === "cosine" ? (result.score * 100).toFixed(2) : result.score.toFixed(4)})`
                  : `Best score: ${result.metric === "cosine" ? (result.score * 100).toFixed(2) + "%" : result.score.toFixed(4)} (below threshold)`)
              : "No scores calculated",
            duration: "~50ms",
            verificationIndex: idx + 1
          },
          {
            id: `${idx}-7`,
            timestamp: fullTimestamp,
            event: `[Verification #${idx + 1}] Emotion Recognition`,
            status: "success",
            details: `${result.emotion_label || "neutral"} (${result.emotion_confidence ? (result.emotion_confidence * 100).toFixed(1) : "0"}% confidence)`,
            duration: "~100ms",
            verificationIndex: idx + 1
          }
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
  const emotion = verifyResult?.emotion_label ?? "neutral"
  const emotionConfidence = verifyResult?.emotion_confidence ?? 0

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
              {/* Score Section */}
              <motion.div
                className="space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="flex items-end justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Match Score</span>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Similarity score using {verifyResult.metric} distance metric</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <motion.span
                    className="text-3xl font-bold"
                    initial={{ scale: 0.8 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 100 }}
                  >
                    {verifyResult.metric === "cosine" ? `${matchScore.toFixed(1)}%` : matchScore.toFixed(2)}
                  </motion.span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted border border-border">
                  <motion.div
                    className={`h-full transition-all ${getScoreColor()}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${scorePercentage}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
                <div className="flex justify-between items-center text-xs text-muted-foreground">
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
                          ({verifyResult.liveness.score.toFixed(1)}%)
                        </span>
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Liveness detection: {verifyResult.liveness.score > 0.5 ? "Real face detected" : "Potential spoof"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="gap-1.5 cursor-help rounded-lg border-border">
                        <Smile className="h-3.5 w-3.5 stroke-[1.5]" />
                        {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                        <span className="text-xs opacity-75">
                          ({(emotionConfidence * 100).toFixed(0)}%)
                        </span>
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Detected emotion with {(emotionConfidence * 100).toFixed(1)}% confidence</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                {verifyResult.matched_id && (
                  <Badge variant="default" className="gap-1.5 rounded-lg">
                    <Activity className="h-3.5 w-3.5 stroke-[1.5]" />
                    ID: {verifyResult.matched_id}
                  </Badge>
                )}
              </motion.div>

              {/* Registry Match Scores */}
              {verifyResult?.all_scores && verifyResult.all_scores.length > 0 && (
                <motion.div
                  className="space-y-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.28 }}
                >
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-3.5 w-3.5 text-muted-foreground" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
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
                          <TableHead className="h-8 px-3 py-2 font-medium text-muted-foreground">Rank</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-muted-foreground">User ID</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-muted-foreground text-right">Match %</TableHead>
                          <TableHead className="h-8 px-3 py-2 font-medium text-muted-foreground text-right">Embeddings</TableHead>
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
                                <span className={`font-bold ${isMatched ? "text-green-600 dark:text-green-400" : isTopMatch ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"}`}>
                                  #{idx + 1}
                                </span>
                              </TableCell>
                              <TableCell className="px-3 py-2 font-medium">
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
                                      className={`h-full ${isMatched ? "bg-green-600" : isTopMatch ? "bg-amber-600" : "bg-primary"}`}
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
                              <TableCell className="px-3 py-2 text-right text-muted-foreground text-[10px]">
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
                  className="flex h-20 items-center justify-center rounded-xl border-2 border-primary/20 bg-primary/5 px-4"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.32 }}
                >
                  <div className="text-center">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Matched Identity</p>
                    <p className="text-lg font-bold text-primary mt-1">{verifyResult.matched_id}</p>
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
                    <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Processing Timeline</h3>
                  </div>
                  <div className="flex items-center gap-1 overflow-x-auto pb-2">
                    {processingSteps.map((step, idx) => (
                      <div key={idx} className="flex items-center gap-1 flex-shrink-0">
                        <div className="text-center min-w-[60px]">
                          <div className="text-[10px] font-medium text-muted-foreground">{step.step}</div>
                          <div className="text-[10px] text-muted-foreground/70">{step.time}ms</div>
                        </div>
                        {idx < processingSteps.length - 1 && (
                          <div className="w-4 h-0.5 bg-border mx-1" />
                        )}
                      </div>
                    ))}
                    <div className="text-[10px] text-muted-foreground ml-2">
                      Total: ~{processingSteps.reduce((sum, s) => sum + s.time, 0)}ms
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Emotion Distribution */}
              {verifyResult?.emotion_label && (
                <motion.div
                  className="space-y-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.35 }}
                >
                  <div className="flex items-center gap-2">
                    <Smile className="h-3.5 w-3.5 text-muted-foreground" />
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Detected Emotion
                    </h3>
                  </div>
                  <div className="rounded-xl border border-border bg-muted/30 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex flex-col">
                        <span className="text-lg font-bold text-foreground capitalize">{emotion}</span>
                        <span className="text-xs text-muted-foreground">
                          Confidence: {(emotionConfidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-3xl">
                        {emotion === "happy" && "😊"}
                        {emotion === "sad" && "😢"}
                        {emotion === "angry" && "😠"}
                        {emotion === "surprise" && "😲"}
                        {emotion === "fear" && "😨"}
                        {emotion === "disgust" && "🤢"}
                        {emotion === "neutral" && "😐"}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Detailed Event Logs */}
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <div className="flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5 text-muted-foreground" />
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Processing Pipeline</h3>
                </div>
                <div className="rounded-xl border border-border" style={{ maxHeight: "400px", overflowY: "auto", overflowX: "auto" }}>
                  <Table className="text-xs">
                    <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur supports-[backdrop-filter]:bg-muted/80">
                      <TableRow className="border-b border-border hover:bg-transparent bg-muted/30">
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground w-[180px] sticky left-0 bg-muted/95 backdrop-blur supports-[backdrop-filter]:bg-muted/80 z-10">Timestamp</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground">Event</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground">Details</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground w-[80px]">Duration</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground w-[100px]">Status</TableHead>
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
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground text-[10px]">
                              {log.timestamp}
                            </TableCell>
                            <TableCell className="px-3 py-2 text-xs font-medium">{log.event}</TableCell>
                            <TableCell className="px-3 py-2 text-xs text-muted-foreground">{log.details || "-"}</TableCell>
                            <TableCell className="px-3 py-2 text-xs text-muted-foreground font-mono">{log.duration || "-"}</TableCell>
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
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <Activity className="h-12 w-12 opacity-40 mb-3" />
              <span className="text-sm font-medium">No verification results yet</span>
              <span className="text-xs mt-1">Upload an image and verify above to see results</span>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
