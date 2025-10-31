"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Info } from "lucide-react"
import { motion } from "framer-motion"

export function EvaluationSection() {
  const metrics = [
    { label: "AUC Score", value: "0.94", tooltip: "Area Under the Receiver Operating Characteristic Curve" },
    { label: "Accuracy", value: "96.2%", tooltip: "Percentage of correct predictions" },
    { label: "Precision", value: "94.8%", tooltip: "True positives / (True positives + False positives)" },
    { label: "Recall", value: "95.1%", tooltip: "True positives / (True positives + False negatives)" },
  ]

  return (
    <motion.div
      className="grid gap-6 lg:grid-cols-2"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      {/* ROC Curve */}
      <Card className="border border-border shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-base font-semibold">ROC Curve</CardTitle>
          <CardDescription className="text-xs">Receiver Operating Characteristic analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <motion.div
            className="flex h-64 items-center justify-center rounded-xl border border-dashed border-border bg-muted/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div className="text-center">
              <p className="text-sm font-medium text-muted-foreground">ROC Curve Visualization</p>
              <p className="text-xs text-muted-foreground mt-1">AUC: 0.94</p>
            </div>
          </motion.div>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card className="border border-border shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-base font-semibold">Performance Metrics</CardTitle>
          <CardDescription className="text-xs">Model evaluation and accuracy scores</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {metrics.map((metric, idx) => (
              <motion.div
                key={metric.label}
                className="flex items-center justify-between rounded-lg border border-border bg-muted/30 px-4 py-3 hover:bg-muted/50 transition-colors"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 + idx * 0.05 }}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{metric.label}</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 stroke-[1.5] text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{metric.tooltip}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <motion.span
                  className="text-lg font-semibold text-foreground"
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 100, delay: 0.35 + idx * 0.05 }}
                >
                  {metric.value}
                </motion.span>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
