# Intelligent Document Processing & Compliance System

A production-grade, distributed ML system designed to ingest messy real-world documents (PDFs, Images, Contracts), perform Optical Character Recognition (OCR), extract entities using Transformers, and assess compliance risks automatically.

## Architecture
* **API**: FastAPI (Async, Type-safe)
* **ML Engine**: Custom Transformer Pipeline (NER + Risk Classification)
* **Orchestration**: Docker & Kubernetes
* **Messaging**: Apache Kafka (Event-driven processing)
* **Caching**: Redis
* **Observability**: Prometheus & Grafana

## Quick Start

### Prerequisites
* Docker & Docker Compose
* Python 3.9+ (for local testing)

### 1. Run the Full Stack
```bash
docker-compose up --build