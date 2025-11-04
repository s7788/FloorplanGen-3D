'use client'

import { useState } from 'react'
import FileUploader from '@/components/FileUploader'
import ModelViewer from '@/components/ModelViewer'
import { useJobStatus } from '@/lib/hooks'
import { JOB_STATUS, API_CONFIG } from '@/lib/constants'

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null)
  const { status, isPolling, error } = useJobStatus(jobId)

  const handleUploadSuccess = (uploadedJobId: string) => {
    setJobId(uploadedJobId)
  }

  // Get model URL from status
  const modelUrl = status?.result_url 
    ? `${API_CONFIG.BASE_URL}${status.result_url}`
    : null

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            FloorplanGen-3D 🏗️
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            將 2D 房屋格局圖自動轉換為可互動的 3D 空間模擬
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Upload */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">上傳平面圖</h2>
            <FileUploader onUploadSuccess={handleUploadSuccess} />
            
            {jobId && status && (
              <div className={`mt-4 p-4 rounded-md border ${
                status.status === JOB_STATUS.COMPLETED ? 'bg-green-50 border-green-200' :
                status.status === JOB_STATUS.FAILED ? 'bg-red-50 border-red-200' :
                status.status === JOB_STATUS.PROCESSING ? 'bg-blue-50 border-blue-200' :
                'bg-gray-50 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium">
                    {status.status === JOB_STATUS.COMPLETED && '✅ 完成'}
                    {status.status === JOB_STATUS.FAILED && '❌ 失敗'}
                    {status.status === JOB_STATUS.PROCESSING && '⏳ 處理中'}
                    {status.status === JOB_STATUS.PENDING && '⏸️ 等待中'}
                  </p>
                  <span className="text-sm font-semibold">{status.progress}%</span>
                </div>
                
                {/* Progress bar */}
                {status.status !== JOB_STATUS.COMPLETED && status.status !== JOB_STATUS.FAILED && (
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${status.progress}%` }}
                    />
                  </div>
                )}
                
                <p className="text-xs text-gray-600">{status.message}</p>
                
                {error && (
                  <p className="text-xs text-red-600 mt-2">{error}</p>
                )}
              </div>
            )}
          </div>

          {/* Right Panel - 3D Viewer */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">3D 預覽</h2>
            <ModelViewer modelUrl={modelUrl} />
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="text-3xl mb-3">🤖</div>
            <h3 className="font-semibold text-lg mb-2">AI 自動辨識</h3>
            <p className="text-sm text-gray-600">
              自動辨識牆體、門窗、空間結構
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="text-3xl mb-3">🎨</div>
            <h3 className="font-semibold text-lg mb-2">3D 生成</h3>
            <p className="text-sm text-gray-600">
              程序化生成可互動的 3D 空間模型
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="text-3xl mb-3">👁️</div>
            <h3 className="font-semibold text-lg mb-2">即時預覽</h3>
            <p className="text-sm text-gray-600">
              360° 旋轉、縮放瀏覽 3D 空間
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
