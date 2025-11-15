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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Emotion Insights</h2>
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

      {/* Anomaly Alerts */}
      {!loading && anomalies.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Attention Required</AlertTitle>
          <AlertDescription>
            {anomalies.length} employee(s) showing concerning emotion patterns. Review details below.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="trends" className="space-y-4">
        <TabsList>
          <TabsTrigger value="trends">
            <TrendingUp className="h-4 w-4 mr-2" />
            Hourly Trends
          </TabsTrigger>
          <TabsTrigger value="departments">
            <Users className="h-4 w-4 mr-2" />
            Department Sentiment
          </TabsTrigger>
          <TabsTrigger value="anomalies">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Anomalies ({anomalies.length})
          </TabsTrigger>
        </TabsList>

        {/* Hourly Trends Tab */}
        <TabsContent value="trends" className="space-y-4">
          <Card className="border-border">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg font-semibold">Daily Emotion Flow</CardTitle>
                  <CardDescription className="text-xs">
                    Workplace sentiment throughout the day
                  </CardDescription>
                </div>
                <div className="flex gap-3 text-xs">
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-2 rounded-full bg-green-500" />
                    <span className="text-muted-foreground">Happiness</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-2 rounded-full bg-red-500" />
                    <span className="text-muted-foreground">Stress</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-2 rounded-full bg-gray-400" />
                    <span className="text-muted-foreground">Neutral</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <Skeleton className="h-[300px] w-full" />
              ) : chartData.length > 0 ? (
                <ChartContainer config={{}} className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted/30" />
                      <XAxis 
                        dataKey="hour" 
                        tick={{ fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ strokeWidth: 0.5 }}
                      />
                      <YAxis 
                        tick={{ fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ strokeWidth: 0.5 }}
                        label={{ value: '%', angle: 0, position: 'top', offset: 10, fontSize: 11 }}
                      />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Line 
                        type="monotone" 
                        dataKey="happiness" 
                        stroke="#22C55E" 
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="Happiness"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="stress" 
                        stroke="#EF4444" 
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="Stress"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="neutral" 
                        stroke="#9CA3AF" 
                        strokeWidth={1.5}
                        dot={{ r: 2 }}
                        name="Neutral"
                        strokeDasharray="5 5"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-sm text-muted-foreground">
                  No emotion data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Department Sentiment Tab */}
        <TabsContent value="departments" className="space-y-4">
          <div className="grid gap-4 grid-cols-1">
            {loading ? (
              Array(4).fill(0).map((_, i) => (
                <Skeleton key={i} className="h-[200px]" />
              ))
            ) : departments.length > 0 ? (
              departments.map(dept => (
                <Card key={dept.department} className="border-border">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-xl font-bold">
                          {dept.department}
                        </CardTitle>
                        <CardDescription className="text-sm mt-1">
                          {dept.total_records} check-ins in last {selectedDays} days
                        </CardDescription>
                      </div>
                      <Badge 
                        variant={
                          dept.wellness_score > 60 ? "default" :
                          dept.wellness_score > 40 ? "secondary" : "destructive"
                        }
                        className="text-lg px-3 py-1 h-auto font-bold"
                      >
                        {dept.wellness_score.toFixed(0)}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-0">
                    {/* Happiness & Stress Bars */}
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1.5">
                          <span className="font-medium text-muted-foreground">😊 Happiness</span>
                          <span className="font-bold text-green-600">
                            {dept.happiness_percentage.toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-3 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-green-500 transition-all duration-500"
                            style={{ width: `${dept.happiness_percentage}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1.5">
                          <span className="font-medium text-muted-foreground">😰 Stress</span>
                          <span className="font-bold text-red-600">
                            {dept.stress_percentage.toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-3 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-red-500 transition-all duration-500"
                            style={{ width: `${dept.stress_percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                    
                    {/* Emotion Distribution Mini Bar */}
                    <div className="pt-2 border-t">
                      <div className="text-xs text-muted-foreground mb-2 font-medium">Emotion Distribution</div>
                      <div className="flex gap-1 h-2 rounded-full overflow-hidden">
                        {Object.entries(dept.emotion_distribution)
                          .sort((a, b) => b[1] - a[1])
                          .map(([emotion, value]) => (
                            <div
                              key={emotion}
                              className="transition-all"
                              style={{
                                width: `${value}%`,
                                backgroundColor: EMOTION_COLORS[emotion] || '#6B7280'
                              }}
                              title={`${emotion}: ${value.toFixed(0)}%`}
                            />
                          ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card className="col-span-full">
                <CardContent className="flex items-center justify-center h-[150px] text-sm text-muted-foreground">
                  No department data available
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* Anomalies Tab */}
        <TabsContent value="anomalies" className="space-y-4">
          {loading ? (
            <Skeleton className="h-48 w-full" />
          ) : anomalies.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {anomalies.map((anomaly, idx) => (
                <Card key={idx} className={`border-2 ${
                  anomaly.severity === "high" ? "border-red-500 bg-red-50 dark:bg-red-950/20" :
                  anomaly.severity === "medium" ? "border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20" :
                  "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
                }`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-3 flex-1">
                        <div className={`p-2.5 rounded-full ${
                          anomaly.severity === "high" ? "bg-red-500" :
                          anomaly.severity === "medium" ? "bg-yellow-500" :
                          "bg-blue-500"
                        }`}>
                          {anomaly.type === "prolonged_negative" ? (
                            <Brain className="h-5 w-5 text-white" />
                          ) : anomaly.type === "high_anger" ? (
                            <AlertTriangle className="h-5 w-5 text-white" />
                          ) : (
                            <Activity className="h-5 w-5 text-white" />
                          )}
                        </div>
                        <div className="flex-1">
                          <CardTitle className="text-lg font-bold">
                            {anomaly.type === "prolonged_negative" ? "⚠️ Prolonged Negative Emotions" :
                             anomaly.type === "high_anger" ? "😡 High Anger Detected" :
                             anomaly.type === "sudden_change" ? "📊 Sudden Emotion Change" :
                             "🔍 Unusual Pattern"}
                          </CardTitle>
                          <CardDescription className="mt-1 text-sm font-medium">
                            User: <span className="font-mono">{anomaly.user_id}</span>
                          </CardDescription>
                        </div>
                      </div>
                      <Badge 
                        variant={getSeverityColor(anomaly.severity)}
                        className="text-xs font-bold uppercase"
                      >
                        {anomaly.severity}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="bg-background/80 rounded-lg p-3 border">
                      <p className="text-sm leading-relaxed">{anomaly.details}</p>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Activity className="h-3.5 w-3.5" />
                      <span>
                        Last detected: {new Date(anomaly.timestamp).toLocaleDateString()} at {new Date(anomaly.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="border-2 border-green-500">
              <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                <div className="bg-green-100 dark:bg-green-950/30 p-4 rounded-full mb-4">
                  <Smile className="h-16 w-16 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-2xl font-bold mb-2">All Clear! 🎉</h3>
                <p className="text-muted-foreground text-lg max-w-md">
                  No concerning emotion patterns detected in the last <span className="font-bold text-foreground">{selectedDays} days</span>.
                </p>
                <p className="text-sm text-muted-foreground mt-3">
                  Your workplace sentiment is healthy and stable.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
