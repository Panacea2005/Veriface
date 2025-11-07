"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { getAttendance, getAttendanceStats, exportAttendance, getUserMetadata, type AttendanceRecord, type AttendanceStats, type VerifyResponse } from "@/lib/api"
import { RefreshCw, Calendar, Users, TrendingUp, Download, Search,  X } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, Legend } from "recharts"

interface AttendanceHistoryProps {
  verifyResult?: VerifyResponse | null
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658']

// Emotion order to ensure consistent color mapping
const EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

// Get color for emotion based on its position in the order
const getEmotionChartColor = (emotion: string): string => {
  const index = EMOTION_ORDER.indexOf(emotion.toLowerCase())
  if (index >= 0) {
    return COLORS[index % COLORS.length]
  }
  // Fallback for unknown emotions
  return COLORS[0]
}

// Get emoji for emotion (matching verification results and webcam section)
const getEmotionEmoji = (emotion: string): string => {
  const emotionLower = emotion.toLowerCase()
  switch (emotionLower) {
    case 'happy':
      return '😊'
    case 'sad':
      return '😢'
    case 'angry':
      return '😠'
    case 'surprise':
      return '😲'
    case 'fear':
      return '😨'
    case 'disgust':
      return '🤢'
    case 'neutral':
      return '😐'
    default:
      return '😐'
  }
}

export function AttendanceHistory({ verifyResult }: AttendanceHistoryProps) {
  const [records, setRecords] = useState<AttendanceRecord[]>([])
  const [stats, setStats] = useState<AttendanceStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [limit, setLimit] = useState(50)
  const [page, setPage] = useState(1)
  const [filterUserId, setFilterUserId] = useState("")
  const [filterDateFrom, setFilterDateFrom] = useState("")
  const [filterDateTo, setFilterDateTo] = useState("")
  const [userMetadata, setUserMetadata] = useState<Record<string, { name?: string; department?: string; email?: string }>>({})
  const { toast } = useToast()
  const lastMatchedIdRef = useRef<string | null>(null)

  const loadAttendance = useCallback(async () => {
    setLoading(true)
    try {
      const params: any = { limit: limit * page }
      if (filterUserId) params.userId = filterUserId
      if (filterDateFrom) params.dateFrom = filterDateFrom
      if (filterDateTo) params.dateTo = filterDateTo

      const [recordsData, statsData, metadata] = await Promise.all([
        getAttendance(params),
        getAttendanceStats({
          userId: filterUserId || undefined,
          dateFrom: filterDateFrom || undefined,
          dateTo: filterDateTo || undefined
        }),
        getUserMetadata()
      ])
      setRecords(recordsData.records)
      setStats(statsData)
      setUserMetadata(metadata)
    } catch (error) {
      console.error("Failed to load attendance:", error)
      toast({
        title: "Failed to load attendance",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }, [limit, page, filterUserId, filterDateFrom, filterDateTo, toast])

  useEffect(() => {
    loadAttendance()
  }, [loadAttendance])

  // Auto-refresh attendance when verification succeeds
  useEffect(() => {
    if (verifyResult?.matched_id && verifyResult.matched_id !== lastMatchedIdRef.current) {
      lastMatchedIdRef.current = verifyResult.matched_id
      const timer = setTimeout(() => {
        loadAttendance()
      }, 800)
      return () => clearTimeout(timer)
    }
  }, [verifyResult?.matched_id, loadAttendance])

  // Auto-refresh stats every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadAttendance()
    }, 30000)
    return () => clearInterval(interval)
  }, [loadAttendance])

  const handleExport = async (format: "csv" | "json") => {
    try {
      const params: any = { format }
      if (filterUserId) params.userId = filterUserId
      if (filterDateFrom) params.dateFrom = filterDateFrom
      if (filterDateTo) params.dateTo = filterDateTo

      const blob = await exportAttendance(params)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `attendance_${new Date().toISOString().split("T")[0]}.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      toast({
        title: "Export successful",
        description: `Attendance data exported as ${format.toUpperCase()}`,
      })
    } catch (error) {
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive"
      })
    }
  }

  const clearFilters = () => {
    setFilterUserId("")
    setFilterDateFrom("")
    setFilterDateTo("")
    setPage(1)
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      const dateStr = date.toLocaleDateString("en-US", { 
        year: "numeric", 
        month: "short", 
        day: "numeric" 
      })
      const timeStr = date.toLocaleTimeString("en-US", { 
        hour: "2-digit", 
        minute: "2-digit",
        second: "2-digit"
      })
      return { date: dateStr, time: timeStr }
    } catch {
      return { date: timestamp, time: "" }
    }
  }

  const getEmotionStyle = (emotion?: string): React.CSSProperties => {
    if (!emotion) return {}
    const chartColor = getEmotionChartColor(emotion)
    // Convert hex to RGB for better opacity control
    const hex = chartColor.replace('#', '')
    const r = parseInt(hex.substring(0, 2), 16)
    const g = parseInt(hex.substring(2, 4), 16)
    const b = parseInt(hex.substring(4, 6), 16)
    return {
      backgroundColor: `rgba(${r}, ${g}, ${b}, 0.1)`,
      color: chartColor,
      borderColor: `rgba(${r}, ${g}, ${b}, 0.3)`,
    } as React.CSSProperties
  }

  // Prepare chart data
  const dailyTrendsData = stats?.daily_trends ? Object.entries(stats.daily_trends)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-14) // Last 14 days
    .map(([date, count]) => ({
      date: new Date(date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      count
    })) : []

  const emotionData = stats?.by_emotion ? Object.entries(stats.by_emotion)
    .filter(([_, count]) => count > 0)
    .map(([emotion, count]) => ({ name: emotion, value: count }))
    .sort((a, b) => {
      // Sort by EMOTION_ORDER to ensure consistent color mapping
      const indexA = EMOTION_ORDER.indexOf(a.name.toLowerCase())
      const indexB = EMOTION_ORDER.indexOf(b.name.toLowerCase())
      if (indexA >= 0 && indexB >= 0) return indexA - indexB
      if (indexA >= 0) return -1
      if (indexB >= 0) return 1
      return a.name.localeCompare(b.name)
    }) : []

  const paginatedRecords = records.slice((page - 1) * limit, page * limit)
  const totalPages = Math.ceil(records.length / limit)

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Records</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_records}</div>
              <p className="text-xs text-foreground/70">Attendance entries</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Unique Users</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.unique_users}</div>
              <p className="text-xs text-foreground/70">Registered users</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Today's Entries</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(() => {
                  const today = new Date().toISOString().split("T")[0]
                  return stats.by_date[today] || 0
                })()}
              </div>
              <p className="text-xs text-foreground/70">Check-ins today</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Check-ins</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.by_type?.["check-in"] || 0}</div>
              <p className="text-xs text-foreground/70">Total check-ins</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Charts */}
      {stats && (
        <Tabs defaultValue="trends" className="w-full">
          <TabsList>
            <TabsTrigger value="trends">Daily Trends</TabsTrigger>
            <TabsTrigger value="emotions">Emotion Distribution</TabsTrigger>
          </TabsList>
          <TabsContent value="trends" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Attendance Trends (Last 14 Days)</CardTitle>
                <CardDescription>Daily attendance count over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{}} className="h-[260px] w-full">
                  <LineChart
                    data={dailyTrendsData}
                    margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
                    width={undefined}
                    height={260}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                    <XAxis
                      dataKey="date"
                      label={{ value: "Date", position: "insideBottom", offset: -10, style: { fontSize: 11 } }}
                      tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }}
                      stroke="hsl(var(--border))"
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }}
                      stroke="hsl(var(--border))"
                    />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Line
                      type="monotone"
                      dataKey="count"
                      stroke="hsl(var(--primary))"
                      strokeWidth={3}
                      activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
                      dot={{ r: 4, fill: "hsl(var(--primary))" }}
                      name="Attendance Count"
                    />
                  </LineChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="emotions" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Emotion Distribution</CardTitle>
                <CardDescription>Distribution of detected emotions</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{}} className="h-[260px] w-full">
                  <PieChart
                    margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
                    width={undefined}
                    height={260}
                  >
                    <Pie
                      data={emotionData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {emotionData.map((entry) => (
                        <Cell key={`cell-${entry.name}`} fill={getEmotionChartColor(entry.name)} />
                      ))}
                    </Pie>
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend
                      verticalAlign="bottom"
                      height={36}
                      iconType="circle"
                      wrapperStyle={{ fontSize: "11px" }}
                    />
                  </PieChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {/* Filters and Export */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Attendance History</CardTitle>
              <CardDescription>Recent attendance records from face verification</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleExport("csv")}
              >
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleExport("json")}
              >
                <Download className="h-4 w-4 mr-2" />
                Export JSON
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={loadAttendance}
                disabled={loading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4 p-4 bg-muted/50 rounded-lg">
            <div className="space-y-2">
              <Label htmlFor="filter-user">User ID</Label>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="filter-user"
                  placeholder="Filter by user..."
                  value={filterUserId}
                  onChange={(e) => {
                    setFilterUserId(e.target.value)
                    setPage(1)
                  }}
                  className="pl-8"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="filter-date-from">Date From</Label>
              <Input
                id="filter-date-from"
                type="date"
                value={filterDateFrom}
                onChange={(e) => {
                  setFilterDateFrom(e.target.value)
                  setPage(1)
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="filter-date-to">Date To</Label>
              <Input
                id="filter-date-to"
                type="date"
                value={filterDateTo}
                onChange={(e) => {
                  setFilterDateTo(e.target.value)
                  setPage(1)
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>&nbsp;</Label>
              <Button
                variant="outline"
                onClick={clearFilters}
                className="w-full"
                disabled={!filterUserId && !filterDateFrom && !filterDateTo}
              >
                <X className="h-4 w-4 mr-2" />
                Clear Filters
              </Button>
            </div>
          </div>

          {/* Records Table */}
          {loading && records.length === 0 ? (
            <div className="text-center py-8 text-foreground/70">Loading attendance records...</div>
          ) : records.length === 0 ? (
            <div className="text-center py-8 text-foreground/70">No attendance records found</div>
          ) : (
            <>
              <div className="space-y-2">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Timestamp</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">User ID</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Name</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Type</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Match Score</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Liveness</th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-foreground/80">Emotion</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedRecords.map((record, idx) => {
                        const { date, time } = formatTimestamp(record.timestamp)
                        const meta = userMetadata[record.user_id] || {}
                        return (
                          <tr key={idx} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
                            <td className="py-3 px-4 text-sm">
                              <div className="font-medium text-foreground">{date}</div>
                              <div className="text-xs text-foreground/70">{time}</div>
                            </td>
                            <td className="py-3 px-4 text-sm font-medium text-foreground">{record.user_id}</td>
                            <td className="py-3 px-4 text-sm text-foreground/80">
                              {meta.name || "-"}
                              {meta.department && (
                                <div className="text-xs text-foreground/60">{meta.department}</div>
                              )}
                            </td>
                            <td className="py-3 px-4 text-sm">
                              <Badge variant={record.type === "check-out" ? "secondary" : "default"} className="font-mono">
                                {record.type === "check-out" ? "OUT" : "IN"}
                              </Badge>
                            </td>
                            <td className="py-3 px-4 text-sm">
                              <Badge variant="secondary" className="font-mono">
                                {(record.match_score * 100).toFixed(1)}%
                              </Badge>
                            </td>
                            <td className="py-3 px-4 text-sm">
                              <Badge variant={record.liveness_score >= 0.5 ? "default" : "destructive"} className="font-mono">
                                {(record.liveness_score * 100).toFixed(1)}%
                              </Badge>
                            </td>
                            <td className="py-3 px-4 text-sm">
                              {record.emotion_label ? (
                                <span className="text-2xl leading-none">
                                  {getEmotionEmoji(record.emotion_label)}
                                </span>
                              ) : (
                                <span className="text-foreground/50 text-xs">N/A</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4">
                  <div className="text-sm text-foreground/70">
                    Showing {(page - 1) * limit + 1} to {Math.min(page * limit, records.length)} of {records.length} records
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      Previous
                    </Button>
                    <div className="text-sm text-foreground/70">
                      Page {page} of {totalPages}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
