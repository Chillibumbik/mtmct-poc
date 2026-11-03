import { useMemo } from 'react'
import { useQueries, type UseQueryResult } from '@tanstack/react-query'
import { api } from '@/api/client'
import { qk } from '@/api/hooks'
import type { TrackBoxes } from '@/api/types.gen'
import { buildCameraIndex, majorityMapping, type CameraIndex } from './frames'
import { WINDOW_S } from './windows'

const combine = (rs: UseQueryResult<TrackBoxes>[]) => ({
  data: rs.map((r) => r.data),
  error: rs.find((r) => r.isError)?.error ?? null,
  loading: rs.some((r) => r.isPending),
})

/**
 * Box windows of every camera fetched so far (each window once, cached for the page's life),
 * indexed per camera, plus the GT → global id majority mapping over everything loaded.
 */
export function useTrackIndex(runId: number, cameras: string[], windows: number[]) {
  const { data, error, loading } = useQueries({
    queries: windows.flatMap((k) =>
      cameras.map((camera) => {
        const q = { camera, t0: k * WINDOW_S, t1: (k + 1) * WINDOW_S }
        return { queryKey: qk.runTracks(runId, q), queryFn: () => api.runTracks(runId, q), staleTime: Infinity }
      }),
    ),
    combine,
  })
  const indices = useMemo(() => {
    const by = new Map<string, TrackBoxes[]>()
    for (const d of data) if (d) by.set(d.camera_id, [...(by.get(d.camera_id) ?? []), d])
    return new Map<string, CameraIndex>([...by].map(([cam, ws]) => [cam, buildCameraIndex(ws)]))
  }, [data])
  const majority = useMemo(() => majorityMapping([...indices.values()].flatMap((i) => i.votes)), [indices])
  return { indices, majority, error, loading }
}
