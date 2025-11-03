# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you get FloorplanGen-3D up and running quickly using Docker.

### Prerequisites

- Docker Desktop or Docker Engine (20.10+)
- Docker Compose (2.0+)
- 4GB RAM available
- 10GB disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/s7788/FloorplanGen-3D.git
cd FloorplanGen-3D
```

### Step 2: Configure Environment

Copy the example environment files:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

> **Note**: The default configuration works out of the box for local development. For production, update the values in these files.

### Step 3: Start All Services

```bash
docker-compose up -d
```

This command will:
1. Start Redis (task queue)
2. Start PostgreSQL (database)
3. Build and start the Backend API
4. Start Celery worker
5. Build and start the Frontend

**First-time startup may take 5-10 minutes** to download images and install dependencies.

### Step 4: Verify Services

Check if all services are running:

```bash
docker-compose ps
```

You should see 5 containers running:
- `floorplangen-redis`
- `floorplangen-postgres`
- `floorplangen-backend`
- `floorplangen-celery`
- `floorplangen-frontend`

### Step 5: Access the Application

Open your browser and navigate to:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Step 6: Upload a Floorplan

1. Visit http://localhost:3000
2. Drag and drop a floorplan image (JPG or PNG)
3. Wait for processing
4. View the generated 3D model in the viewer

---

## 🔧 Troubleshooting

### Services Not Starting

Check the logs for a specific service:

```bash
docker-compose logs backend
docker-compose logs frontend
```

### Port Already in Use

If ports 3000 or 8000 are already in use, edit `docker-compose.yml` and change the port mappings.

### Frontend Not Loading

Wait 2-3 minutes for npm dependencies to install on first startup. Check logs:

```bash
docker-compose logs -f frontend
```

### Backend API Errors

Ensure Redis and PostgreSQL are running:

```bash
docker-compose ps redis postgres
```

### Reset Everything

To start fresh, remove all containers and volumes:

```bash
docker-compose down -v
docker-compose up -d --build
```

---

## 📝 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Upload a floorplan
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/floorplan.jpg"

# Check job status
curl http://localhost:8000/api/v1/status/{job_id}
```

### Using the Browser

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

---

## 🛠️ Development Mode

### Running Backend Locally

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at http://localhost:8000

### Running Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at http://localhost:3000

---

## 📚 Next Steps

- Read the [Development Guide](docs/DEVELOPMENT.md) for detailed setup
- Check the [API Documentation](docs/API.md) for API details
- Review the [Project Roadmap](PROJECT_ROADMAP.md) for upcoming features
- Explore the code structure in the README.md

---

## 🆘 Need Help?

- Check the [Documentation](docs/)
- Open an issue on GitHub
- Review the logs: `docker-compose logs`

---

## 🛑 Stopping the Application

```bash
# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything including volumes
docker-compose down -v
```
