'use client'

import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei'
import { MESSAGES } from '@/lib/constants'

interface ModelViewerProps {
  modelUrl: string | null
}

function Scene() {
  return (
    <>
      <PerspectiveCamera makeDefault position={[5, 5, 5]} />
      <OrbitControls enableDamping dampingFactor={0.05} />
      
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.3} />
      
      <Grid args={[20, 20]} cellSize={1} cellColor="#6b7280" sectionColor="#374151" />
      
      {/* Placeholder box - in production, load actual model */}
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[2, 1, 3]} />
        <meshStandardMaterial color="#94a3b8" />
      </mesh>
    </>
  )
}

export default function ModelViewer({ modelUrl }: ModelViewerProps) {
  return (
    <div className="w-full h-[500px] bg-gray-100 rounded-lg overflow-hidden">
      {modelUrl ? (
        <Canvas>
          <Suspense fallback={null}>
            <Scene />
          </Suspense>
        </Canvas>
      ) : (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
              />
            </svg>
            <p className="mt-2 text-sm text-gray-500">
              {MESSAGES.UPLOAD_PLACEHOLDER}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
