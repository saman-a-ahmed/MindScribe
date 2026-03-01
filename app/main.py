"""
FastAPI application for MindScribe
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import Query
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db, init_db
from app.repository import JournalRepository
from app.config import settings
from src.analyzer import JournalAnalyzer

# Initialize analyzer (singleton)
analyzer = None

def get_analyzer() -> JournalAnalyzer:
    """Get or create analyzer instance"""
    global analyzer
    if analyzer is None:
        analyzer = JournalAnalyzer(
            emotion_model_path=settings.EMOTION_MODEL_PATH,
            distortion_model_path=settings.DISTORTION_MODEL_PATH,
            emotion_threshold=settings.EMOTION_THRESHOLD,
            distortion_threshold=settings.DISTORTION_THRESHOLD,
            verbose=True
        )
    return analyzer

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered journaling application with emotion and cognitive distortion analysis"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and verify models on startup"""
    init_db()
    # Verify analyzer can load
    try:
        get_analyzer()
        print("✓ Analyzer initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Analyzer initialization issue: {e}")

# Pydantic models for request/response
class JournalEntryCreate(BaseModel):
    text: str
    user_id: Optional[str] = "default"

class EmotionAnalysisResponse(BaseModel):
    emotion: str
    probability: float

class DistortionAnalysisResponse(BaseModel):
    label: str
    name: str
    description: str
    probability: float

class AnalysisResult(BaseModel):
    emotions: Optional[dict] = None
    distortions: Optional[dict] = None
    insights: List[str] = []
    recommendations: List[str] = []

class JournalEntryResponse(BaseModel):
    id: int
    text: str
    timestamp: datetime
    user_id: str
    analysis: Optional[AnalysisResult] = None

    class Config:
        from_attributes = True

class JournalEntryListResponse(BaseModel):
    entries: List[JournalEntryResponse]
    total: int
    skip: int
    limit: int

class TrendDataPoint(BaseModel):
    date: str
    emotion: str
    avg_probability: float
    count: int

class EmotionStatistics(BaseModel):
    emotion: str
    avg_probability: float
    count: int
    max_probability: float

class StatisticsResponse(BaseModel):
    emotions: List[EmotionStatistics]
    total_entries: int

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    analyzer_status = get_analyzer().get_status()
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "models": analyzer_status
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    analyzer_status = get_analyzer().get_status()
    if not analyzer_status['ready']:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Models not loaded"}
        )
    return {"status": "healthy", "models": analyzer_status}

# Journal entry endpoints
@app.post("/api/entries", response_model=JournalEntryResponse, status_code=status.HTTP_201_CREATED)
async def create_entry(
    entry: JournalEntryCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new journal entry with automatic analysis
    """
    try:
        # Analyze the entry
        analysis_result = get_analyzer().analyze(entry.text)
        
        # Create entry in database
        repo = JournalRepository(db)
        journal_entry = repo.create_entry(
            text=entry.text,
            user_id=entry.user_id,
            analysis_result=analysis_result
        )
        
        # Prepare response
        response = JournalEntryResponse(
            id=journal_entry.id,
            text=journal_entry.text,
            timestamp=journal_entry.timestamp,
            user_id=journal_entry.user_id,
            analysis=AnalysisResult(
                emotions=analysis_result.get('emotions'),
                distortions=analysis_result.get('distortions'),
                insights=analysis_result.get('insights', []),
                recommendations=analysis_result.get('recommendations', [])
            )
        )
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating entry: {str(e)}"
        )

@app.get("/api/entries", response_model=JournalEntryListResponse)
async def get_entries(
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Get journal entries with pagination and date filtering
    """
    repo = JournalRepository(db)
    entries = repo.get_entries(
        user_id=user_id,
        skip=skip,
        limit=limit,
        start_date=start_date,
        end_date=end_date
    )
    
    # Convert to response models
    entry_responses = []
    for entry in entries:
        # Re-analyze to get full analysis (or could store in DB)
        analysis_result = get_analyzer().analyze(entry.text)
        entry_responses.append(JournalEntryResponse(
            id=entry.id,
            text=entry.text,
            timestamp=entry.timestamp,
            user_id=entry.user_id,
            analysis=AnalysisResult(
                emotions=analysis_result.get('emotions'),
                distortions=analysis_result.get('distortions'),
                insights=analysis_result.get('insights', []),
                recommendations=analysis_result.get('recommendations', [])
            )
        ))
    
    total = repo.get_entries(user_id=user_id).__len__()  # Simplified - should use count query
    
    return JournalEntryListResponse(
        entries=entry_responses,
        total=total,
        skip=skip,
        limit=limit
    )

@app.get("/api/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_entry(
    entry_id: int,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Get a specific journal entry by ID
    """
    repo = JournalRepository(db)
    entry = repo.get_entry(entry_id, user_id)
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found"
        )
    
    # Re-analyze to get full analysis
    analysis_result = get_analyzer().analyze(entry.text)
    
    return JournalEntryResponse(
        id=entry.id,
        text=entry.text,
        timestamp=entry.timestamp,
        user_id=entry.user_id,
        analysis=AnalysisResult(
            emotions=analysis_result.get('emotions'),
            distortions=analysis_result.get('distortions'),
            insights=analysis_result.get('insights', []),
            recommendations=analysis_result.get('recommendations', [])
        )
    )

@app.delete("/api/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entry(
    entry_id: int,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Delete a journal entry
    """
    repo = JournalRepository(db)
    deleted = repo.delete_entry(entry_id, user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found"
        )
    
    return None

# Trends and statistics endpoints
@app.get("/api/entries/trends", response_model=List[TrendDataPoint])
async def get_trends(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    emotion: Optional[str] = None,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Get emotional trends over time
    """
    repo = JournalRepository(db)
    trends = repo.get_emotion_trends(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        emotion=emotion
    )
    
    return [TrendDataPoint(**trend) for trend in trends]

@app.get("/api/entries/stats", response_model=StatisticsResponse)
async def get_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Get aggregated emotion statistics
    """
    repo = JournalRepository(db)
    stats = repo.get_emotion_statistics(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return StatisticsResponse(
        emotions=[EmotionStatistics(**emotion) for emotion in stats['emotions']],
        total_entries=stats['total_entries']
    )

@app.get("/api/distortions/stats")
async def get_distortion_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """
    Get aggregated distortion statistics
    """
    repo = JournalRepository(db)
    stats = repo.get_distortion_statistics(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return stats

# Analysis endpoint (without saving)
class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text without saving to database
    """
    try:
        result = get_analyzer().analyze(request.text)
        return AnalysisResult(
            emotions=result.get('emotions'),
            distortions=result.get('distortions'),
            insights=result.get('insights', []),
            recommendations=result.get('recommendations', [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
