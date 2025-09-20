from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl
import httpx
import uuid
import logging
from typing import Optional
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent Backend",
    description="Backend service for AI article processing workflow",
    version="1.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ArticleRequest(BaseModel):
    email: EmailStr
    article_url: HttpUrl

class ArticleResponse(BaseModel):
    session_id: str
    message: str
    status: str

class WebhookPayload(BaseModel):
    email: str
    article_url: str
    session_id: str

class StatusResponse(BaseModel):
    session_id: str
    status: str
    processed_at: Optional[datetime] = None

# In-memory storage for session tracking (use Redis/Database in production)
session_store = {}

# Configuration - n8n webhook URL
# Your production webhook URL is set as default
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "https://fahim09.app.n8n.cloud/webhook/8eea811c-ba70-4f5a-8d33-765019d67bb0")

@app.on_event("startup")
async def startup_event():
    """Startup event to validate configuration"""
    logger.info(f"Using n8n webhook URL: {N8N_WEBHOOK_URL}")
    logger.info("FastAPI AI Agent Backend started successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Agent Backend is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "n8n_webhook_url": N8N_WEBHOOK_URL
    }

async def send_to_n8n(payload: WebhookPayload) -> bool:
    """Send payload to n8n webhook asynchronously"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                N8N_WEBHOOK_URL,
                json=payload.dict(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Successfully sent data to n8n for session {payload.session_id}")
            return True
    except httpx.RequestError as e:
        logger.error(f"Request error when sending to n8n: {e}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error when sending to n8n: {e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when sending to n8n: {e}")
        return False

@app.post("/process-article", response_model=ArticleResponse)
async def process_article(request: ArticleRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint to process article
    Generates session ID and forwards to n8n webhook
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Store session info
        session_store[session_id] = {
            "email": str(request.email),
            "article_url": str(request.article_url),
            "status": "processing",
            "created_at": datetime.utcnow(),
            "processed_at": None
        }
        
        # Create payload for n8n
        webhook_payload = WebhookPayload(
            email=str(request.email),
            article_url=str(request.article_url),
            session_id=session_id
        )
        
        # Send to n8n in background
        background_tasks.add_task(process_in_background, webhook_payload)
        
        logger.info(f"Created session {session_id} for {request.email}")
        
        return ArticleResponse(
            session_id=session_id,
            message="Article processing started. You will receive an email when complete.",
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error processing article request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_in_background(payload: WebhookPayload):
    """Background task to send data to n8n and update status"""
    success = await send_to_n8n(payload)
    
    # Update session status
    if payload.session_id in session_store:
        session_store[payload.session_id]["status"] = "sent_to_n8n" if success else "failed"
        if success:
            session_store[payload.session_id]["processed_at"] = datetime.utcnow()

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get processing status for a session"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = session_store[session_id]
    return StatusResponse(
        session_id=session_id,
        status=session_data["status"],
        processed_at=session_data.get("processed_at")
    )

@app.get("/sessions")
async def list_sessions():
    """List all sessions (for debugging - remove in production)"""
    return {"sessions": list(session_store.keys()), "total": len(session_store)}

@app.post("/webhook/n8n-callback")
async def n8n_callback(payload: dict):
    """
    Optional callback endpoint for n8n to report completion status
    n8n can call this endpoint when processing is complete
    """
    try:
        session_id = payload.get("session_id")
        status = payload.get("status", "completed")
        
        if session_id and session_id in session_store:
            session_store[session_id]["status"] = status
            session_store[session_id]["processed_at"] = datetime.utcnow()
            logger.info(f"Updated status for session {session_id}: {status}")
            
        return {"message": "Status updated", "session_id": session_id}
    
    except Exception as e:
        logger.error(f"Error in n8n callback: {e}")
        raise HTTPException(status_code=500, detail="Callback processing failed")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (cleanup endpoint)"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del session_store[session_id]
    return {"message": f"Session {session_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)