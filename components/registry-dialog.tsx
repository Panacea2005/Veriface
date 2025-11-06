"use client"

import React from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { fetchRegistry, RegistryResponse, deleteRegistryEmbedding, deleteRegistryUser } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

// Prefer green tones: use chart-2 exclusively to avoid yellow hues
const palette = ["var(--chart-2)"]

function colorForUser(userId: string): string {
  let h = 0
  for (let i = 0; i < userId.length; i++) {
    h = ((h << 5) - h) + userId.charCodeAt(i)
    h |= 0
  }
  const v = palette[Math.abs(h) % palette.length]
  return `hsl(${v})`
}

interface RegistryDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function RegistryDialog({ open, onOpenChange }: RegistryDialogProps) {
  const [data, setData] = React.useState<RegistryResponse | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [query, setQuery] = React.useState("")
  const [selectedUser, setSelectedUser] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState(false)

  React.useEffect(() => {
    if (!open) return
    let mounted = true
    setLoading(true)
    // Fetch with raw vectors for numeric rendering
    fetchRegistry({ project: "pca2d", includeVectors: true, limitPerUser: 5000 })
      .then((res) => {
        if (!mounted) return
        setData(res)
        if (res.users.length > 0) setSelectedUser(res.users[0])
      })
      .catch((e) => { if (mounted) setError(String(e?.message || e)) })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [open])

  const filteredUsers = React.useMemo(() => {
    if (!data) return []
    if (!query) return data.users
    const q = query.toLowerCase()
    return data.users.filter(u => u.toLowerCase().includes(q))
  }, [data, query])

  // no plot when rendering raw vectors

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[98vw] max-w-[98vw] md:max-w-[98vw] lg:max-w-[98vw] xl:max-w-[98vw] p-0 overflow-hidden max-h-[92vh]">
        <DialogHeader className="px-6 pt-6">
          <DialogTitle>Registry Explorer</DialogTitle>
          <DialogDescription>Browse identities and read the exact embedding vectors per identity.</DialogDescription>
        </DialogHeader>

        <div className="px-6 pb-6">
          {error && <div className="text-sm text-red-500 mb-3">{error}</div>}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Left: list */}
            <div className="md:col-span-1 border rounded-lg">
              <div className="p-3 border-b">
                <Input placeholder="Search user…" value={query} onChange={(e) => setQuery(e.target.value)} />
              </div>
              <ScrollArea className="h-[420px]">
                <ul className="divide-y">
                  {filteredUsers.map((u) => (
                    <li key={u} className={`flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-muted ${selectedUser === u ? "bg-muted" : ""}`} onClick={() => setSelectedUser(u)}>
                      <div className="flex items-center gap-2">
                        <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: colorForUser(u) }} />
                        <span className="text-sm font-medium">{u}</span>
                      </div>
                      <Badge variant="secondary">{data?.counts[u] ?? 0}</Badge>
                    </li>
                  ))}
                  {filteredUsers.length === 0 && (
                    <li className="px-3 py-4 text-sm text-muted-foreground">No users</li>
                  )}
                </ul>
              </ScrollArea>
              {selectedUser && (
                <div className="p-3 border-t flex items-center justify-between">
                  <div className="text-xs text-muted-foreground">
                    Showing embeddings for <span className="font-medium">{selectedUser}</span>
                  </div>
                  <Button size="sm" variant="destructive" disabled={busy}
                    onClick={async () => {
                      if (!selectedUser) return
                      if (!confirm(`Delete ALL embeddings for '${selectedUser}'?`)) return
                      setBusy(true)
                      try {
                        await deleteRegistryUser(selectedUser)
                        // refresh
                        const res = await fetchRegistry({ project: "pca2d", includeVectors: true, limitPerUser: 5000 })
                        setData(res)
                        setSelectedUser(null)
                      } catch (e: any) {
                        alert(e?.message || String(e))
                      } finally {
                        setBusy(false)
                      }
                    }}>Delete user</Button>
                </div>
              )}
            </div>

            {/* Right: raw vectors for selected user (numeric view) */}
            <div className="md:col-span-2 border rounded-lg">
              <div className="flex items-center justify-between p-3 border-b">
                <div className="text-sm font-medium">Embedding Vectors</div>
                <div className="text-xs text-muted-foreground">
                  {selectedUser ? (
                    <>
                      <Badge className="mr-2" style={{ background: colorForUser(selectedUser), color: "#fff" }}>{selectedUser}</Badge>
                      {(data?.vectors?.[selectedUser]?.length || 0)} vectors
                    </>
                  ) : (
                    <>Select a user to view vectors</>
                  )}
                </div>
              </div>
              <ScrollArea className="h-[70vh]">
                <div className="p-4 grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(380px, 1fr))" }}>
                  {selectedUser && (data?.vectors?.[selectedUser] ?? []).map((vec, i) => (
                    <div key={i} className="border rounded-md bg-white/50 dark:bg-black/30">
                      <div className="px-3 py-2 text-xs font-medium bg-muted/60 border-b flex items-center justify-between">
                        <span>Embedding #{i + 1}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">{vec.length} dims</span>
                          <Button size="sm" variant="outline" disabled={busy}
                            onClick={async () => {
                              if (!selectedUser) return
                              if (!confirm(`Delete embedding #${i + 1} for '${selectedUser}'?`)) return
                              setBusy(true)
                              try {
                                await deleteRegistryEmbedding(selectedUser, i)
                                const res = await fetchRegistry({ project: "pca2d", includeVectors: true, limitPerUser: 5000 })
                                setData(res)
                              } catch (e: any) {
                                alert(e?.message || String(e))
                              } finally {
                                setBusy(false)
                              }
                            }}>Delete</Button>
                        </div>
                      </div>
                      <div className="px-3 py-2 font-mono text-[11px] leading-relaxed text-foreground/90 whitespace-pre-wrap break-words">
                        [{vec.map(v => Number.parseFloat(String(v)).toFixed(4)).join(", ")}]
                      </div>
                    </div>
                  ))}
                  {selectedUser && (data?.vectors?.[selectedUser]?.length ?? 0) === 0 && (
                    <div className="text-sm text-muted-foreground">No vectors available for this user.</div>
                  )}
                  {!selectedUser && (
                    <div className="text-sm text-muted-foreground">Choose a user from the left panel.</div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}


