import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'FloorplanGen-3D',
  description: '將 2D 房屋格局圖自動轉換為可互動的 3D 空間模擬',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-TW">
      <body className="bg-gray-50">{children}</body>
    </html>
  )
}
