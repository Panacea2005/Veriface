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
          <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-4">
            {loading ? (
              Array(4).fill(0).map((_, i) => (
                <Skeleton key={i} className="h-[110px]" />
              ))
            ) : departments.length > 0 ? (
              departments.map(dept => (
                <Card key={dept.department} className="border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center justify-between">
                      <span className="truncate">{dept.department}</span>
                      <Badge 
                        variant={
                          dept.wellness_score > 60 ? "default" :
                          dept.wellness_score > 40 ? "secondary" : "destructive"
                        }
                        className="text-[10px] px-1.5 py-0 h-5 ml-2"
                      >
                        {dept.wellness_score.toFixed(0)}
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-[10px]">
                      {dept.total_records} check-ins
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-1.5 pt-0">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Happy</span>
                      <span className="font-medium text-green-600">
                        {dept.happiness_percentage.toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Stress</span>
                      <span className="font-medium text-red-600">
                        {dept.stress_percentage.toFixed(0)}%
                      </span>
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
            <div className="space-y-3">
              {anomalies.map((anomaly, idx) => (
                <Alert key={idx} variant={anomaly.severity === "high" ? "destructive" : "default"}>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle className="flex items-center gap-2">
                    User: {anomaly.user_id}
                    <Badge variant={getSeverityColor(anomaly.severity)}>
                      {anomaly.severity} severity
                    </Badge>
                  </AlertTitle>
                  <AlertDescription>
                    <p className="font-medium">
                      {anomaly.type === "prolonged_negative" ? "Prolonged Negative Emotions" :
                       anomaly.type === "high_anger" ? "High Anger Detected" :
                       anomaly.type === "sudden_change" ? "Sudden Emotion Change" :
                       "Unusual Pattern"}
                    </p>
                    <p className="text-sm mt-1">{anomaly.details}</p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Last detected: {new Date(anomaly.timestamp).toLocaleString()}
                    </p>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center h-48 text-center">
                <Smile className="h-12 w-12 text-green-500 mb-4" />
                <h3 className="text-lg font-semibold">All Clear!</h3>
                <p className="text-muted-foreground mt-2">
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
