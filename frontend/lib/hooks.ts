/**
 * Custom hook for polling job status
 */
import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { API_CONFIG, JOB_STATUS } from './constants'

interface JobStatusData {
  job_id: string
  status: string
  progress: number
  message: string
  result_url?: string
  error?: string
}

export function useJobStatus(jobId: string | null) {
  const [status, setStatus] = useState<JobStatusData | null>(null)
  const [isPolling, setIsPolling] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    if (!jobId) return

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await axios.get(`${apiUrl}/api/v1/status/${jobId}`)
      const data = response.data as JobStatusData

      setStatus(data)

      // Stop polling if completed or failed
      if (data.status === JOB_STATUS.COMPLETED || data.status === JOB_STATUS.FAILED) {
        setIsPolling(false)
      }

      if (data.error) {
        setError(data.error)
      }
    } catch (err: any) {
      console.error('Status fetch error:', err)
      setError(err.response?.data?.detail || 'Failed to fetch status')
      setIsPolling(false)
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId) {
      setIsPolling(false)
      return
    }

    // Start polling
    setIsPolling(true)
    setError(null)

    // Initial fetch
    fetchStatus()

    // Set up polling interval
    const interval = setInterval(fetchStatus, API_CONFIG.POLL_INTERVAL)

    return () => {
      clearInterval(interval)
    }
  }, [jobId, fetchStatus])

  return { status, isPolling, error }
}
