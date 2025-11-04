# FloorplanGen-3D 🏗️

> 將 2D 房屋格局圖自動轉換為可互動的 3D 空間模擬

## 🎯 專案概述

FloorplanGen-3D 是一個創新的網頁服務，讓使用者上傳 2D 平面圖（JPG/PNG），透過 AI 分析與程序化 3D 生成技術，自動產出可互動的 3D 空間模擬場景。

### 核心功能

- ✅ 2D 格局圖上傳（JPG, PNG）
- 🤖 AI 自動辨識牆體、門窗、空間
- 🏷️ AI 自動標記空間類型（客廳、廚房、臥室、浴室）
- 🎨 程序化生成 3D 模型
- 🪑 自動化 3D 傢俱佈置
- 👁️ 網頁端可互動 3D 瀏覽器（360° 旋轉、縮放）

## 🏗️ 系統架構

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Frontend  │ ───► │  Backend API │ ───► │  Task Queue     │
│  (Next.js)  │      │  (FastAPI)   │      │  (Redis/Celery) │
└─────────────┘      └──────────────┘      └─────────────────┘
                              │                      │
                              ▼                      ▼
                     ┌─────────────┐       ┌─────────────────┐
                     │  S3 Storage │       │  AI/CV Service  │
                     └─────────────┘       └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ 3D Gen Service  │
                                            │   (Blender)     │
                                            └─────────────────┘
```

## 🛠️ 技術棧

### Frontend
- **Framework**: Next.js 14 + TypeScript
- **3D Rendering**: Three.js + @react-three/fiber + @react-three/drei
- **UI**: Tailwind CSS
- **File Upload**: react-dropzone

### Backend
- **API**: FastAPI (Python 3.11+)
- **Task Queue**: Celery + Redis
- **Storage**: AWS S3 / Google Cloud Storage
- **Database**: PostgreSQL (for job status tracking)

### AI/CV Service
- **Framework**: PyTorch
- **CV Library**: OpenCV
- **Models**: U-Net (segmentation), ResNet (classification)

### 3D Generation
- **Engine**: Blender Python API (bpy)
- **Format**: glTF 2.0 (.gltf, .glb)

## 📦 快速開始

### 🚀 5 分鐘快速啟動

使用 Docker Compose 快速啟動所有服務：

```bash
git clone https://github.com/s7788/FloorplanGen-3D.git
cd FloorplanGen-3D
docker-compose up -d
```

訪問 http://localhost:3000 開始使用！

詳細步驟請參閱 [QUICKSTART.md](./QUICKSTART.md)

### 前置需求
- Docker & Docker Compose (推薦)
- Node.js 18+ (本地前端開發)
- Python 3.11+ (本地後端開發)

### 環境變數配置

複製 `.env.example` 為 `.env`：

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

> 預設配置適用於本地開發環境

## 📚 開發階段

詳細開發路線圖請參閱 [PROJECT_ROADMAP.md](./PROJECT_ROADMAP.md)

### Phase 1: MVP ✅
- ✅ 基礎前端介面
- ✅ 檔案上傳 API
- ✅ 簡化版 CV 處理（規則基礎）
- ✅ 基礎 3D 生成

### Phase 2: AI 增強 ✅
- ✅ 深度學習模型架構
- ✅ U-Net 牆體分割
- ✅ ResNet 房間類型分類
- ✅ Faster R-CNN 門窗偵測
- [ ] 模型訓練（需要數據集）

### Phase 3: 進階功能
- [ ] 程序化傢俱佈置
- [ ] 多樓層支援
- [ ] 自訂材質與顏色

## 📖 文件

- [API 文件](./docs/API.md)
- [AI 增強技術文件](./docs/AI_ENHANCEMENT.md)
- [資料結構說明](./docs/DATA_STRUCTURE.md)
- [開發指南](./docs/DEVELOPMENT.md)
- [部署指南](./docs/DEPLOYMENT.md)

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

## 📄 授權

MIT License

## 👨‍💻 作者

[@s7788](https://github.com/s7788)

---

**專案狀態**: 🚧 開發中 | **當前版本**: 0.2.0-beta | **最後更新**: 2025-11-04