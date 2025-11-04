'use client'

import { useState } from 'react'
import FileUploader from '@/components/FileUploader'
import ModelViewer from '@/components/ModelViewer'

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [modelUrl, setModelUrl] = useState<string | null>(null)

  const handleUploadSuccess = (uploadedJobId: string) => {
    setJobId(uploadedJobId)
    // In production, poll for job status and update modelUrl when ready
    console.log('Upload successful, job ID:', uploadedJobId)
  }

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
            
            {jobId && (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
                <p className="text-sm text-blue-800">
                  <span className="font-medium">Job ID:</span> {jobId}
                </p>
                <p className="text-xs text-blue-600 mt-1">
                  處理中... 請稍候
                </p>
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
