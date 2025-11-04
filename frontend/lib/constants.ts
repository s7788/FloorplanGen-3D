/**
 * Application constants and configuration
 */

export const MESSAGES = {
  DRAG_DROP: '拖曳檔案至此或點擊上傳',
  DRAG_DROP_ACTIVE: '放開以上傳檔案',
  FILE_SUPPORT: '支援 JPG、PNG 格式，最大 10MB',
  UPLOADING: '上傳中...',
  UPLOAD_PLACEHOLDER: '上傳平面圖以開始生成 3D 模型',
  UPLOAD_ERROR_DEFAULT: '上傳失敗，請稍後再試',
}

export const API_CONFIG = {
  ACCEPTED_FORMATS: {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
  },
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  POLL_INTERVAL: 2000, // 2 seconds
  BASE_URL: typeof window !== 'undefined' 
    ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
    : 'http://localhost:8000',
}

export const JOB_STATUS = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const
