"use client"

import { useState } from "react"
import { CheckCircle2, Smile, Eye } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { motion } from "framer-motion"

export function ResultsCard() {
  const [matchScore] = useState(87.5)
  const [threshold] = useState(80)
  const [isLive] = useState(true)
  const [emotion] = useState("neutral")
  const [isLoading] = useState(false)

  const logs = [
    { timestamp: "14:32:45", event: "Face detected", status: "success" },
    { timestamp: "14:32:46", event: "Liveness check", status: "success" },
    { timestamp: "14:32:47", event: "Feature extraction", status: "success" },
    { timestamp: "14:32:48", event: "Matching", status: "success" },
  ]

  const isMatched = matchScore >= threshold
  const scorePercentage = (matchScore / 100) * 100

  const getScoreColor = () => {
    if (matchScore >= threshold) return "bg-green-600"
    if (matchScore >= threshold - 10) return "bg-amber-600"
    return "bg-red-600"
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
          <CardDescription className="text-xs">Real-time analysis and matching</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          {isLoading ? (
            <>
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-8 w-full rounded-lg" />
            </>
          ) : (
            <>
              {/* Score Section */}
              <motion.div
                className="space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="flex items-end justify-between">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Match Score</span>
                  <motion.span
                    className="text-3xl font-bold"
                    initial={{ scale: 0.8 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 100 }}
                  >
                    {matchScore}%
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
                        <span className="cursor-help">Threshold: {threshold}%</span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Minimum score required for a match</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <motion.span
                    className={`font-medium ${isMatched ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400"}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    {isMatched ? "✓ Match" : "○ No Match"}
                  </motion.span>
                </div>
              </motion.div>

              {/* Badges Section */}
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
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Liveness detection status</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="gap-1.5 cursor-help rounded-lg border-border">
                        <Smile className="h-3.5 w-3.5 stroke-[1.5]" />
                        {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Detected emotion</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </motion.div>

              {/* Thumbnail */}
              <motion.div
                className="flex h-24 items-center justify-center rounded-xl border border-border bg-muted"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
              >
                <span className="text-xs text-muted-foreground font-medium">Captured face thumbnail</span>
              </motion.div>

              {/* Logs Table */}
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.35 }}
              >
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Event Log</h3>
                <div className="overflow-hidden rounded-xl border border-border">
                  <Table className="text-xs">
                    <TableHeader>
                      <TableRow className="border-b border-border hover:bg-transparent bg-muted/30">
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground">Time</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground">Event</TableHead>
                        <TableHead className="h-9 px-3 py-2 font-medium text-muted-foreground">Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {logs.map((log, idx) => (
                        <motion.tr
                          key={idx}
                          className="border-b border-border last:border-b-0 hover:bg-muted/50 transition-colors"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.4 + idx * 0.05 }}
                        >
                          <TableCell className="px-3 py-2 font-mono text-muted-foreground text-xs">
                            {log.timestamp}
                          </TableCell>
                          <TableCell className="px-3 py-2 text-xs">{log.event}</TableCell>
                          <TableCell className="px-3 py-2">
                            <Badge
                              variant="outline"
                              className="bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300 border-green-200 dark:border-green-800 rounded-md gap-1"
                            >
                              <CheckCircle2 className="h-3 w-3 stroke-[1.5]" />
                              {log.status}
                            </Badge>
                          </TableCell>
                        </motion.tr>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </motion.div>
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
