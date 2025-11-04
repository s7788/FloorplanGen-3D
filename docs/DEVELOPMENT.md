# Development Guide

## Prerequisites

- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- Git

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/s7788/FloorplanGen-3D.git
cd FloorplanGen-3D
```

### 2. Set Up Environment Variables

Copy the example environment files:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Edit these files with your configuration.

### 3. Start with Docker Compose (Recommended)

```bash
docker-compose up -d
```

This will start:
- Backend API (http://localhost:8000)
- Frontend (http://localhost:3000)
- Redis (localhost:6379)
- PostgreSQL (localhost:5432)
- Celery worker

### 4. Manual Setup (Alternative)

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
FloorplanGen-3D/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── core/        # Core configuration
│   │   ├── models/      # Data models
│   │   └── services/    # Business logic
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            # Next.js frontend
│   ├── app/            # Next.js app directory
│   ├── components/     # React components
│   ├── lib/           # Utilities
│   ├── Dockerfile
│   └── package.json
├── docs/              # Documentation
├── docker-compose.yml
└── README.md
```

## Development Workflow

### Backend Development

1. Make changes to Python files in `backend/app/`
2. The server auto-reloads with `--reload` flag
3. Test endpoints using curl or Postman:
   ```bash
   curl http://localhost:8000/health
   ```

### Frontend Development

1. Make changes to TypeScript/React files in `frontend/`
2. Next.js auto-reloads in development mode
3. View changes at http://localhost:3000

## Testing

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

## Common Tasks

### View Backend Logs

```bash
docker-compose logs -f backend
```

### View Frontend Logs

```bash
docker-compose logs -f frontend
```

### Restart Services

```bash
docker-compose restart
```

### Stop All Services

```bash
docker-compose down
```

### Rebuild After Code Changes

```bash
docker-compose up -d --build
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

## Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use, modify `docker-compose.yml` to use different ports.

### Database Connection Issues

Ensure PostgreSQL is running:
```bash
docker-compose ps postgres
```

### Redis Connection Issues

Ensure Redis is running:
```bash
docker-compose ps redis
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request
