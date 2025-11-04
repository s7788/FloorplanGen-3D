# API Documentation

## Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### POST /upload

Upload a 2D floorplan image for processing.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (JPG or PNG, max 10MB)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "File uploaded successfully. Processing queued.",
  "upload_time": "2025-11-03T15:00:00Z"
}
```

**Status Codes:**
- `200 OK`: File uploaded successfully
- `400 Bad Request`: Invalid file type or size
- `500 Internal Server Error`: Server error

---

### GET /status/{job_id}

Get the processing status of a job.

**Request:**
- Method: `GET`
- Path Parameter: `job_id` (string)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "progress": 50,
  "message": "Processing floorplan...",
  "result_url": null,
  "error": null,
  "created_at": "2025-11-03T15:00:00Z",
  "updated_at": "2025-11-03T15:01:00Z"
}
```

**Status Values:**
- `pending`: Job is queued
- `processing`: Job is being processed
- `completed`: Job completed successfully
- `failed`: Job failed

**Status Codes:**
- `200 OK`: Status retrieved successfully
- `404 Not Found`: Job not found

---

## Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Examples

### Upload a floorplan

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@floorplan.jpg"
```

### Check job status

```bash
curl http://localhost:8000/api/v1/status/{job_id}
```
