# Deployment Guide for OCR Document System

This guide covers deployment options for the OCR Document Processing System with MongoDB integration.

## Local Deployment

### Prerequisites

1. Python 3.8+ installed
2. MongoDB installed and running
3. Sufficient disk space for model files (~5GB)
4. CUDA-compatible GPU recommended but not required

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/document-ocr-mongodb.git
   cd document-ocr-mongodb
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB=ocr_db
   MONGODB_USER_COLLECTION=users
   MONGODB_OCR_COLLECTION=ocr_data
   MONGODB_EMBEDDINGS_COLLECTION=embeddings
   ```

5. **Start the API server**
   ```bash
   python -m uvicorn api.routes:app --host 0.0.0.0 --port 8000
   ```

6. **Access the web interface**
   Open http://localhost:8000/ in your browser.

## Docker Deployment

### Prerequisites

1. Docker installed
2. Docker Compose (optional, for multi-container setup)
3. MongoDB accessible (can be containerized or external)

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t document-ocr-mongodb .
   ```

2. **Run with Docker**
   ```bash
   docker run -p 8000:8000 \
     -e MONGODB_URI=mongodb://host.docker.internal:27017 \
     -e MONGODB_DB=ocr_db \
     document-ocr-mongodb
   ```

   Note: Use `host.docker.internal` to access MongoDB running on host machine.

### Using Docker Compose

1. **Create a `docker-compose.yml` file**
   ```yaml
   version: '3'
   services:
     ocr-api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - MONGODB_URI=mongodb://mongodb:27017
         - MONGODB_DB=ocr_db
       volumes:
         - ./sample_data:/app/sample_data
       depends_on:
         - mongodb
     
     mongodb:
       image: mongo:latest
       ports:
         - "27017:27017"
       volumes:
         - mongodb_data:/data/db
   
   volumes:
     mongodb_data:
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Access the web interface**
   Open http://localhost:8000/ in your browser.

## Production Deployment

For production environments, consider these additional steps:

### 1. Use Gunicorn as WSGI Server

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.routes:app -b 0.0.0.0:8000
```

### 2. Set Up Reverse Proxy

Using Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. MongoDB Security

1. Enable authentication in MongoDB
2. Use connection string with credentials:
   ```
   MONGODB_URI=mongodb://username:password@hostname:27017/ocr_db
   ```
3. Consider MongoDB Atlas for managed database service

### 4. HTTPS/SSL Setup

Using Let's Encrypt with Certbot and Nginx:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Cloud Deployment

### AWS Deployment

1. **Containerize the application** using Docker
2. **Push to Amazon ECR**
3. **Deploy using ECS** or **EKS**
4. **Use MongoDB Atlas** or **DocumentDB**

Example AWS CLI commands:
```bash
# Create ECR repository
aws ecr create-repository --repository-name document-ocr-mongodb

# Push Docker image
aws ecr get-login-password | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker tag document-ocr-mongodb:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/document-ocr-mongodb:latest
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/document-ocr-mongodb:latest
```

### Google Cloud Platform

1. **Build and push to Google Container Registry**
2. **Deploy to Google Cloud Run** or **GKE**
3. **Use MongoDB Atlas** or **Cosmos DB**

Example GCP commands:
```bash
# Build using Cloud Build
gcloud builds submit --tag gcr.io/project-id/document-ocr-mongodb

# Deploy to Cloud Run
gcloud run deploy document-ocr --image gcr.io/project-id/document-ocr-mongodb \
  --platform managed --allow-unauthenticated \
  --set-env-vars="MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/ocr_db"
```

## Monitoring and Logging

1. **Prometheus** for metrics collection
2. **Grafana** for visualization
3. **Sentry** for error tracking (already in requirements.txt)
4. **ELK Stack** or **CloudWatch** for log aggregation

## Scaling Considerations

1. **Horizontal scaling**: Deploy multiple API instances behind a load balancer
2. **MongoDB scaling**: Use sharding for large datasets
3. **Model optimization**: Consider quantization for faster inference
4. **Caching**: Implement Redis for caching frequent requests
5. **Queue processing**: Use Celery or RabbitMQ for handling high-volume processing

## Backup and Disaster Recovery

1. **MongoDB backups**: Set up regular MongoDB backups
2. **Document storage**: Consider AWS S3 or other object storage for original images
3. **Infrastructure as Code**: Use Terraform or CloudFormation to define infrastructure 