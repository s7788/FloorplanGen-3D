# FloorplanGen-3D Project Roadmap

## Overview

This document outlines the development phases for FloorplanGen-3D, from MVP to full production-ready application.

---

## Phase 1: MVP (Minimum Viable Product) ✅

**Status**: Implemented  
**Duration**: Week 1-2

### Goals
Establish basic functionality with rule-based processing and simple 3D generation.

### Features
- ✅ Basic frontend interface (Next.js + TypeScript)
  - File upload component with react-dropzone
  - 3D viewer placeholder using Three.js
  - Responsive UI with Tailwind CSS
- ✅ File upload API (FastAPI)
  - POST /api/v1/upload endpoint
  - File validation (JPG/PNG, max 10MB)
  - Job ID generation
- ✅ Simplified CV processing (rule-based)
  - OpenCV-based wall detection
  - Basic room segmentation
  - Contour detection
- ✅ Basic 3D generation
  - glTF structure generation
  - Wall and floor geometry
  - Simple material assignment
- ✅ Docker setup
  - docker-compose.yml configuration
  - Backend and frontend Dockerfiles
  - Redis and PostgreSQL services
- ✅ Environment configuration
  - .env.example files
  - Configuration management

### Tech Stack
- Frontend: Next.js 14, TypeScript, Tailwind CSS, Three.js
- Backend: FastAPI, Python 3.11+
- CV: OpenCV, NumPy
- Infrastructure: Docker, Redis, PostgreSQL

---

## Phase 2: AI Enhancement 🔄

**Status**: Planned  
**Duration**: Week 3-6

### Goals
Replace rule-based processing with deep learning models for better accuracy.

### Features
- [ ] Deep learning model training
  - Collect and prepare training dataset
  - Data augmentation pipeline
- [ ] U-Net wall segmentation
  - Train wall detection model
  - Integrate with processing pipeline
- [ ] Room type classification
  - ResNet-based classifier
  - Classify rooms (bedroom, kitchen, bathroom, living room, etc.)
- [ ] Door and window detection
  - Object detection model (YOLO/Faster R-CNN)
  - Accurate opening identification
- [ ] Model optimization
  - Model quantization
  - Inference optimization

### Tech Stack
- ML Framework: PyTorch
- Model Serving: TorchServe or ONNX Runtime
- Training: GPU-accelerated (CUDA)

---

## Phase 3: Advanced Features 🔮

**Status**: Planned  
**Duration**: Week 7-10

### Goals
Add advanced features for better user experience and model quality.

### Features
- [ ] Procedural furniture placement
  - Room-specific furniture sets
  - Intelligent placement algorithms
  - Collision detection
- [ ] Multi-floor support
  - Multiple floor upload
  - Floor linking and navigation
  - Staircase detection
- [ ] Custom materials and colors
  - Material library
  - Color palette selection
  - Texture mapping
- [ ] Interactive editing
  - Manual wall adjustment
  - Room label editing
  - Furniture repositioning
- [ ] Export options
  - glTF/GLB download
  - FBX export
  - OBJ export

### Tech Stack
- 3D: Blender Python API (bpy)
- Procedural Generation: Custom algorithms
- Storage: AWS S3 / Google Cloud Storage

---

## Phase 4: Production & Scale 🚀

**Status**: Planned  
**Duration**: Week 11-14

### Goals
Prepare for production deployment and scale to handle multiple users.

### Features
- [ ] User authentication
  - OAuth2 integration
  - User accounts and profiles
- [ ] Payment integration
  - Subscription plans
  - Pay-per-use options
- [ ] Performance optimization
  - CDN integration
  - Caching strategy
  - Database optimization
- [ ] Monitoring and logging
  - Application metrics
  - Error tracking (Sentry)
  - Usage analytics
- [ ] CI/CD pipeline
  - Automated testing
  - Deployment automation
  - Environment management
- [ ] Production deployment
  - Cloud infrastructure (AWS/GCP)
  - Load balancing
  - Auto-scaling

### Tech Stack
- Auth: OAuth2, JWT
- Payments: Stripe
- Monitoring: Prometheus, Grafana
- CI/CD: GitHub Actions
- Cloud: AWS or Google Cloud Platform

---

## Future Enhancements 💡

### Long-term Features
- Mobile app (React Native)
- AR/VR visualization
- Real-time collaboration
- AI-powered design suggestions
- Energy efficiency analysis
- Cost estimation tools
- Integration with CAD software

---

## Success Metrics

### Phase 1 (MVP)
- Upload and process basic floorplans
- Generate simple 3D models
- Basic visualization working

### Phase 2 (AI Enhancement)
- 90%+ accuracy in wall detection
- 85%+ accuracy in room classification
- Processing time < 30 seconds

### Phase 3 (Advanced Features)
- Support 5+ furniture types per room
- Multi-floor plans with 3+ floors
- User satisfaction score > 4.0/5.0

### Phase 4 (Production)
- 99.9% uptime
- Handle 1000+ concurrent users
- Average response time < 2 seconds

---

## Version History

- v0.1.0-alpha (Current) - Phase 1 MVP Implementation
- v0.2.0-beta (Planned) - Phase 2 AI Enhancement
- v0.3.0-rc (Planned) - Phase 3 Advanced Features
- v1.0.0 (Planned) - Phase 4 Production Release
