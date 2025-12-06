# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from datetime import datetime
import json
import logging
import time
import os
from contextlib import asynccontextmanager
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer
import aiofiles

from ml_pipeline.models import ComplianceNERModel, RiskClassifier, SemanticSearchEngine
from data_ingestion.pipeline import DocumentIngestor

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis & Kafka Globals
redis_client = None
kafka_producer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, kafka_producer
    
    # 1. Initializing Redis with retry logic or fallback
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        logger.info("Redis client initialized.")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")

    # Initializing Kafka in Safe Mode
    # If Kafka is not reachable, it won't crash, it just will not send messages.
    try:
        kafka_producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        )
        await kafka_producer.start()
        logger.info("Kafka producer initialized.")
    except Exception as e:
        logger.warning(f"Kafka connection failed (running in offline mode): {e}")
        kafka_producer = None
    
    # Initializing ML Models
    logger.info("Loading ML Models...")
    app.state.ner_model = ComplianceNERModel()
    app.state.risk_classifier = RiskClassifier()
    app.state.search_engine = SemanticSearchEngine()
    # Initializing Ingestor with empty config
    app.state.document_ingestor = DocumentIngestor({})
    logger.info("ML Models loaded successfully.")
    
    yield
    
    # Shuting down
    if redis_client:
        await redis_client.close()
    if kafka_producer:
        await kafka_producer.stop()

app = FastAPI(
    title="Compliance Monitoring API",
    description="Production-grade document processing and compliance monitoring system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic Models
class DocumentUpload(BaseModel):
    filename: str
    file_type: str = Field(..., pattern="^(pdf|docx|txt|image|unknown)$")
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    text: str
    search_type: str = Field("semantic", pattern="^(semantic|keyword|hybrid)$")
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(10, ge=1, le=100)

class ComplianceCheckRequest(BaseModel):
    document_id: str
    regulations: List[str] = Field(default=["SOX", "GDPR", "PCI-DSS"])

class BatchProcessRequest(BaseModel):
    document_ids: List[str]
    priority: str = Field("normal", pattern="^(low|normal|high|critical)$")

# Dependencies
async def rate_limit(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Mock Rate Limiter"""
    # In a real demo, if Redis fails, it bypass rate limiting
    if not redis_client:
        return credentials.credentials

    try:
        key = f"rate_limit:{credentials.credentials}"
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, 60)
        if current > 100:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception:
        pass # It fails to open if Redis is down
        
    return credentials.credentials

# Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "compliance-monitoring",
        "models_loaded": hasattr(app.state, 'ner_model')
    }

@app.post("/api/v1/documents/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    user_token: str = Depends(rate_limit)
):
    """
    Upload a document for processing (Ingestion Layer)
    """
    try:
        content = await file.read()
        
        # Saving to temp (simulating cloud storage upload)
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        # Ingesting
        raw_doc = app.state.document_ingestor.ingest_document(temp_path)
        
        # Async Processing Trigger (Kafka)
        if kafka_producer:
            message = {
                "document_id": raw_doc.id,
                "filename": raw_doc.filename,
                "timestamp": datetime.utcnow().isoformat()
            }
            await kafka_producer.send_and_wait(
                "document-processing",
                json.dumps(message).encode()
            )
            msg_status = "Queued in Kafka"
        else:
            msg_status = "Kafka unavailable - Processed Synchronously (Demo Mode)"

        # Cleaning up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "document_id": raw_doc.id,
            "status": "processing",
            "message": msg_status,
            "extracted_metadata": raw_doc.metadata
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search")
async def semantic_search(
    query: QueryRequest,
    user_token: str = Depends(rate_limit)
):
    """
    Semantic search across processed documents (Retrieval Layer)
    """
    try:
        results = app.state.search_engine.search_similar(
            query.text,
            k=query.limit
        )
        return {
            "query": query.text,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/compliance/check")
async def compliance_check(
    check_request: ComplianceCheckRequest,
    user_token: str = Depends(rate_limit)
):
    """
    Run compliance checks against regulations (Reasoning Layer)
    """
    # Getting document details (Mock)
    # Extracting Entities
    entities = app.state.ner_model.extract_entities(check_request.document_id)
    
    # Assessing Risk
    risk = app.state.risk_classifier.classify_risk(check_request.document_id, entities['transformer_entities'])
    
    return {
        "document_id": check_request.document_id,
        "risk_assessment": risk,
        "entities_detected": entities,
        "compliance_status": "FAIL" if risk['risk_level'] == "HIGH" else "PASS"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)