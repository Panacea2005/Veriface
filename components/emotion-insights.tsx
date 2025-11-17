"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Smile, TrendingUp, Users, AlertTriangle, Activity, Brain } from "lucide-react"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

const EMOTION_COLORS: Record<string, string> = {
  'angry': '#EF4444',      // Red
  'disgust': '#84CC16',    // Lime green
  'fear': '#8B5CF6',       // Purple
  'happy': '#F59E0B',      // Amber
  'sad': '#3B82F6',        // Blue
  'surprise': '#EC4899',   // Pink
  'neutral': '#6B7280',    // Gray
}

interface HourlyTrend {
  hour: number
  total_records: number
  emotions: Record<string, number>
  dominant_emotion: string
}

interface UserProfile {
  user_id: string
  period: string
  total_records: number
  emotion_distribution: Record<string, number>
  dominant_emotion: string
  wellness_score: number
  concern_flags: string[]
  recommendation: string
}

interface DepartmentSentiment {
  department: string
  total_records: number
  happiness_percentage: number
  stress_percentage: number
  wellness_score: number
  emotion_distribution: Record<string, number>
}

interface Anomaly {
  type: string
  user_id: string
  severity: "low" | "medium" | "high"
  details: string
  timestamp: string
}

export function EmotionInsights() {
  const [loading, setLoading] = useState(true)
  const [hourlyTrends, setHourlyTrends] = useState<HourlyTrend[]>([])
  const [departments, setDepartments] = useState<DepartmentSentiment[]>([])
  const [anomalies, setAnomalies] = useState<Anomaly[]>([])
  const [selectedDays, setSelectedDays] = useState(7)

  useEffect(() => {
    loadInsights()
  }, [selectedDays])

  const loadInsights = async () => {
    setLoading(true)
    try {
      const [trendsRes, deptRes, anomalyRes] = await Promise.all([
        fetch(`http://localhost:8000/emotion-analytics/trends/hourly?days=${selectedDays}`),
        fetch(`http://localhost:8000/emotion-analytics/department/sentiment?days=${selectedDays}`),
        fetch(`http://localhost:8000/emotion-analytics/anomalies?days=${selectedDays}`)
      ])

      // Check for errors
      if (!trendsRes.ok) {
        console.error("Trends API error:", await trendsRes.text())
        setHourlyTrends([])
      } else {
        const trendsData = await trendsRes.json()
        setHourlyTrends(trendsData.hourly_trends || [])
      }

      if (!deptRes.ok) {
        console.error("Department API error:", await deptRes.text())
        setDepartments([])
      } else {
        const deptData = await deptRes.json()
        setDepartments(deptData.departments || [])
      }

      if (!anomalyRes.ok) {
        console.error("Anomaly API error:", await anomalyRes.text())
        setAnomalies([])
      } else {
        const anomalyData = await anomalyRes.json()
        setAnomalies(anomalyData.anomalies || [])
      }
    } catch (error) {
      console.error("Failed to load emotion insights:", error)
      // Set empty data on error
      setHourlyTrends([])
      setDepartments([])
      setAnomalies([])
    } finally {
      setLoading(false)
    }
  }

  const getWellnessColor = (score: number) => {
    if (score >= 70) return "text-green-600 dark:text-green-400"
    if (score >= 50) return "text-yellow-600 dark:text-yellow-400"
    return "text-red-600 dark:text-red-400"
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "destructive"
      case "medium": return "default"
      default: return "secondary"
    }
  }

  // Format hourly trends for line chart (cleaner)
  const chartData = hourlyTrends.map(trend => {
    const dominant = Object.entries(trend.emotions).reduce((a, b) => a[1] > b[1] ? a : b, ['neutral', 0])
    return {
      hour: `${trend.hour}h`,
      happiness: (trend.emotions.happy || 0) + (trend.emotions.surprise || 0),
      stress: (trend.emotions.angry || 0) + (trend.emotions.fear || 0) + (trend.emotions.sad || 0),
      neutral: trend.emotions.neutral || 0,
      dominant: dominant[0]
    }
  })

  const totalRecords = hourlyTrends.reduce((sum, trend) => sum + trend.total_records, 0)
  const mostActiveHour = chartData.reduce((best, current) => {
    if (!best || current.happiness + current.stress > best.happiness + best.stress) {
      return current
    }
    return best
  }, chartData[0])
  const dominantEmotionOverall = (() => {
    const aggregate: Record<string, number> = {}
    hourlyTrends.forEach(trend => {
      Object.entries(trend.emotions).forEach(([emotion, value]) => {
        aggregate[emotion] = (aggregate[emotion] || 0) + value
      })
    })
    return Object.entries(aggregate).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "neutral"
  })()
  const avgWellness = departments.length
    ? departments.reduce((sum, dept) => sum + dept.wellness_score, 0) / departments.length
    : null
  const anomalyCount = anomalies.length
  const topDepartments = [...departments].sort((a, b) => b.wellness_score - a.wellness_score).slice(0, 5)
  const deptChartData = topDepartments.map(dept => {
    const happiness = Number(dept.happiness_percentage.toFixed(1))
    const stress = Number(dept.stress_percentage.toFixed(1))
    const neutral = Math.max(0, Number((100 - happiness - stress).toFixed(1)))
    return {
      name: dept.department,
      happiness,
      stress,
      neutral,
      wellness: dept.wellness_score.toFixed(0)
    }
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Emotion Insights</h2>
          <p className="text-muted-foreground">
            Workplace sentiment analysis and mental health monitoring
          </p>
        </div>
        <div className="flex gap-2">
          <Badge
            variant={selectedDays === 7 ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setSelectedDays(7)}
          >
            7 Days
          </Badge>
          <Badge
            variant={selectedDays === 14 ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setSelectedDays(14)}
          >
            14 Days
          </Badge>
          <Badge
            variant={selectedDays === 30 ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setSelectedDays(30)}
          >
            30 Days
          </Badge>
        </div>
      </div>

      {/* Summary tiles */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="border border-border rounded-lg shadow-sm bg-background">
          <CardHeader className="pb-2">
            <CardDescription className="text-[10px] uppercase tracking-[0.25em]">Total Records</CardDescription>
            <CardTitle className="text-2xl font-semibold">{totalRecords.toLocaleString()}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Captured in the last {selectedDays} days
          </CardContent>
        </Card>
        <Card className="border border-border rounded-lg shadow-sm bg-background">
          <CardHeader className="pb-2">
            <CardDescription className="text-[10px] uppercase tracking-[0.25em]">Dominant Emotion</CardDescription>
            <CardTitle className="text-2xl font-semibold capitalize">{dominantEmotionOverall}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Detected most frequently
          </CardContent>
        </Card>
        <Card className="border border-border rounded-lg shadow-sm bg-background">
          <CardHeader className="pb-2">
            <CardDescription className="text-[10px] uppercase tracking-[0.25em]">Avg. Wellness</CardDescription>
            <CardTitle className="text-2xl font-semibold">{avgWellness ? avgWellness.toFixed(1) : "—"}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Across {departments.length || 0} departments
          </CardContent>
        </Card>
        <Card className="border border-border rounded-lg shadow-sm bg-background">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <div>
              <CardDescription className="text-[10px] uppercase tracking-[0.25em]">Anomalies</CardDescription>
              <CardTitle className="text-2xl font-semibold">{anomalyCount}</CardTitle>
            </div>
            {anomalyCount > 0 && (
              <Badge variant="destructive" className="font-semibold">{anomalyCount > 9 ? "9+" : anomalyCount}</Badge>
            )}
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Flagged emotion events
          </CardContent>
        </Card>
      </div>

      {/* Anomaly banner */}
      {!loading && anomalies.length > 0 && (
        <Alert variant="destructive" className="border border-destructive/40 bg-destructive/5">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle className="font-semibold">Attention required</AlertTitle>
          <AlertDescription className="text-sm">
            {anomalies.length} team member(s) displaying unusual emotion patterns. Review anomaly tab for details.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="trends" className="space-y-4">
        <TabsList className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <TabsTrigger value="trends" className="gap-2 rounded-xl border border-transparent data-[state=active]:border-border data-[state=active]:bg-background shadow-sm">
            <TrendingUp className="h-4 w-4 mr-2" />
            Hourly Trends
          </TabsTrigger>
          <TabsTrigger value="departments" className="gap-2 rounded-xl border border-transparent data-[state=active]:border-border data-[state=active]:bg-background shadow-sm">
            <Users className="h-4 w-4 mr-2" />
            Department Sentiment
          </TabsTrigger>
          <TabsTrigger value="anomalies" className="gap-2 rounded-xl border border-transparent data-[state=active]:border-border data-[state=active]:bg-background shadow-sm">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Anomalies ({anomalies.length})
          </TabsTrigger>
        </TabsList>

        {/* Hourly Trends Tab */}
        <TabsContent value="trends" className="space-y-4">
          <div className="rounded-xl border border-border bg-background shadow-sm p-6 space-y-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-muted-foreground">Hourly Overview</p>
                <h3 className="text-2xl font-semibold">Daily Emotion Flow</h3>
                <p className="text-sm text-muted-foreground">Live sentiment mix across the workday</p>
              </div>
              {!loading && mostActiveHour && (
            <div className="grid grid-cols-3 gap-3 text-[11px] text-muted-foreground">
              <div className="rounded-lg border border-border/50 px-3 py-2 bg-background">
                    <p className="tracking-[0.3em] uppercase text-[9px] mb-1">Peak</p>
                    <p className="text-base font-semibold text-foreground">{mostActiveHour.hour}</p>
                  </div>
              <div className="rounded-lg border border-border/50 px-3 py-2 bg-background">
                    <p className="tracking-[0.3em] uppercase text-[9px] mb-1">Happiness</p>
                    <p className="text-base font-semibold text-green-500">{mostActiveHour.happiness.toFixed(0)}%</p>
                  </div>
              <div className="rounded-lg border border-border/50 px-3 py-2 bg-background">
                    <p className="tracking-[0.3em] uppercase text-[9px] mb-1">Stress</p>
                    <p className="text-base font-semibold text-red-500">{mostActiveHour.stress.toFixed(0)}%</p>
                  </div>
                </div>
              )}
            </div>
            <div className="flex gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <span className="inline-flex h-2 w-2 rounded-full bg-green-500" />
                Happiness
              </div>
              <div className="flex items-center gap-1.5">
                <span className="inline-flex h-2 w-2 rounded-full bg-red-500" />
                Stress
              </div>
              <div className="flex items-center gap-1.5">
                <span className="inline-flex h-2 w-2 rounded-full bg-gray-400" />
                Neutral
              </div>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              {loading ? (
                <Skeleton className="h-[320px] w-full rounded-2xl" />
              ) : chartData.length > 0 ? (
                <ChartContainer config={{}} className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <XAxis dataKey="hour" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} label={{ value: '%', position: 'insideLeft', offset: -8, fontSize: 11 }} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Line type="monotone" dataKey="happiness" stroke="#22C55E" strokeWidth={2.5} dot={{ r: 0 }} activeDot={{ r: 4 }} name="Happiness" />
                      <Line type="monotone" dataKey="stress" stroke="#EF4444" strokeWidth={2.5} dot={{ r: 0 }} activeDot={{ r: 4 }} name="Stress" />
                      <Line type="monotone" dataKey="neutral" stroke="#9CA3AF" strokeWidth={1.5} strokeDasharray="6 6" dot={{ r: 0 }} activeDot={{ r: 3 }} name="Neutral" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              ) : (
                <div className="flex items-center justify-center h-[320px] text-sm text-muted-foreground">
                  No emotion data available
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Department Sentiment Tab */}
        <TabsContent value="departments" className="space-y-4">
          <div className="space-y-4">
            {loading
              ? Array(4).fill(0).map((_, i) => <Skeleton key={i} className="h-[320px]" />)
              : departments.length > 0
                ? departments.map(dept => {
                    const deptName = (dept.department || "").toLowerCase() === "unknown"
                      ? "Swinburne University of Technology"
                      : dept.department
                    return (
                      <Card key={dept.department} className="border border-border rounded-xl shadow-sm p-6 space-y-6">
                      <div className="flex items-start justify-between">
                        <div>
                          <p className="text-xs uppercase tracking-[0.4em] text-muted-foreground">Department</p>
                          <h3 className="text-xl font-semibold">{deptName}</h3>
                          <p className="text-sm text-muted-foreground">
                            {dept.total_records} check-ins • {selectedDays}-day window
                          </p>
                        </div>
                        <Badge variant="outline" className="text-sm px-3 py-1 font-semibold">
                          {dept.wellness_score.toFixed(0)}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Happiness</p>
                          <p className="text-3xl font-bold text-green-500">{dept.happiness_percentage.toFixed(0)}%</p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Stress</p>
                          <p className="text-3xl font-bold text-red-500">{dept.stress_percentage.toFixed(0)}%</p>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="h-3 rounded-full bg-muted/70 overflow-hidden">
                          <div className="h-full bg-green-500" style={{ width: `${dept.happiness_percentage}%` }} />
                        </div>
                        <div className="h-3 rounded-full bg-muted/70 overflow-hidden">
                          <div className="h-full bg-red-500" style={{ width: `${dept.stress_percentage}%` }} />
                        </div>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground mb-2">Distribution</p>
                        <div className="space-y-2">
                          <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
                            {["happy","sad","angry","fear","neutral"].map(key => (
                              <div key={key} className="flex items-center gap-1.5">
                                <span className="inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: EMOTION_COLORS[key] }} />
                                {key}
                              </div>
                            ))}
                          </div>
                          <div className="flex gap-1 h-3 rounded-full overflow-hidden border border-border/50">
                            {Object.entries(dept.emotion_distribution)
                              .sort((a, b) => b[1] - a[1])
                              .map(([emotion, value]) => (
                                <div
                                  key={emotion}
                                  className="h-full"
                                  style={{
                                    width: `${value}%`,
                                    backgroundColor: EMOTION_COLORS[emotion] || '#6B7280'
                                  }}
                                />
                              ))}
                          </div>
                        </div>
                      </div>
                      </Card>
                    )
                  })
                : (
                    <Card>
                      <CardContent className="flex items-center justify-center h-[180px] text-sm text-muted-foreground">
                        No department data available
                      </CardContent>
                    </Card>
                  )}
          </div>
        </TabsContent>

        {/* Anomalies Tab */}
        <TabsContent value="anomalies" className="space-y-4">
          {loading ? (
            <Skeleton className="h-[320px] w-full" />
          ) : anomalies.length > 0 ? (
            anomalies.map((anomaly, idx) => (
              <Card key={idx} className="border border-border rounded-xl shadow-sm p-6 h-[320px] flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.4em] text-muted-foreground">Anomaly</p>
                    <h4 className="text-2xl font-semibold">{anomaly.type.replace(/_/g, " ")}</h4>
                  </div>
                  <Badge variant={getSeverityColor(anomaly.severity)} className="font-semibold uppercase tracking-wide">
                    {anomaly.severity}
                  </Badge>
                </div>
                <p className="text-base text-foreground leading-relaxed flex-1 mt-4">{anomaly.details}</p>
                <div className="flex flex-wrap gap-4 text-xs text-muted-foreground border-t border-border/70 pt-3">
                  <span>User {anomaly.user_id}</span>
                  <span>{new Date(anomaly.timestamp).toLocaleString()}</span>
                </div>
              </Card>
            ))
          ) : (
            <Card className="border border-border rounded-xl shadow-sm h-[320px] flex items-center justify-center">
              <CardContent className="text-center space-y-3">
                <Smile className="h-12 w-12 text-green-500 mx-auto" />
                <p className="text-lg font-semibold text-foreground">All clear!</p>
                <p className="text-sm text-muted-foreground">
                  No concerning emotion patterns detected in the last {selectedDays} days.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
