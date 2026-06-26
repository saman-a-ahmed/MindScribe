"""
FastAPI application for MindScribe.

Exposes the journaling API: create/list/read/delete entries (with emotion and
cognitive-distortion analysis), one-off text analysis, and aggregated
trends/statistics. Heavy ML inference runs only when *creating* or *analyzing*
text; read endpoints rebuild their analysis from data already stored in the
database (see ``entry_to_response``).
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.repository import JournalRepository
from app.config import settings
from app.models import JournalEntry
from src.analyzer import JournalAnalyzer
from src.feedback import derive_insights, derive_recommendations

# ---------------------------------------------------------------------------
# Analyzer singleton
# ---------------------------------------------------------------------------
analyzer: Optional[JournalAnalyzer] = None


def get_analyzer() -> JournalAnalyzer:
    """Get or lazily create the shared :class:`JournalAnalyzer` instance."""
    global analyzer
    if analyzer is None:
        analyzer = JournalAnalyzer(
            emotion_model_path=settings.EMOTION_MODEL_PATH,
            distortion_model_path=settings.DISTORTION_MODEL_PATH,
            emotion_threshold=settings.EMOTION_THRESHOLD,
            distortion_threshold=settings.DISTORTION_THRESHOLD,
            verbose=True,
        )
    return analyzer


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database and warm up the analyzer on startup."""
    init_db()
    try:
        get_analyzer()
        print("✓ Analyzer initialized successfully")
    except Exception as e:  # noqa: BLE001 - startup should not crash the app
        print(f"⚠️  Warning: Analyzer initialization issue: {e}")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered journaling application with emotion and cognitive distortion analysis",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class JournalEntryCreate(BaseModel):
    text: str
    user_id: Optional[str] = "default"


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


class AnalyzeRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _analysis_from_entry(entry: JournalEntry) -> Dict[str, Any]:
    """
    Rebuild an analysis-result dictionary from an entry's stored rows.

    Avoids re-running the ML models on every read by reusing the emotions and
    distortions persisted when the entry was created.
    """
    emotions_detected = [
        {"emotion": e.emotion, "probability": e.probability}
        for e in sorted(entry.emotions, key=lambda x: x.probability, reverse=True)
    ]
    distortions_detected = [
        {
            "label": d.distortion_type,
            "name": d.distortion_name,
            "description": d.description or "",
            "probability": d.probability,
        }
        for d in sorted(entry.distortions, key=lambda x: x.probability, reverse=True)
    ]

    emotions = {
        "detected": emotions_detected,
        "top_emotions": emotions_detected,
        "primary_emotion": emotions_detected[0] if emotions_detected else None,
    }
    distortions = {
        "detected": distortions_detected,
        "top_distortions": distortions_detected,
        "has_distortions": len(distortions_detected) > 0,
        "count": len(distortions_detected),
    }
    return {"emotions": emotions, "distortions": distortions}


def entry_to_response(entry: JournalEntry) -> JournalEntryResponse:
    """Convert a stored ``JournalEntry`` into the API response model."""
    analysis = _analysis_from_entry(entry)
    return JournalEntryResponse(
        id=entry.id,
        text=entry.text,
        timestamp=entry.timestamp,
        user_id=entry.user_id,
        analysis=AnalysisResult(
            emotions=analysis["emotions"],
            distortions=analysis["distortions"],
            insights=derive_insights(analysis),
            recommendations=derive_recommendations(analysis),
        ),
    )


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint - API information and model status."""
    analyzer_status = get_analyzer().get_status()
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "models": analyzer_status,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    analyzer_status = get_analyzer().get_status()
    if not analyzer_status["ready"]:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Models not loaded"},
        )
    return {"status": "healthy", "models": analyzer_status}


# ---------------------------------------------------------------------------
# Journal entry endpoints
#
# NOTE: Static sub-paths (``/api/entries/trends``, ``/api/entries/stats``) MUST
# be declared before the dynamic ``/api/entries/{entry_id}`` route, otherwise
# FastAPI matches "trends"/"stats" as an integer ``entry_id`` and returns 422.
# ---------------------------------------------------------------------------
@app.post("/api/entries", response_model=JournalEntryResponse, status_code=status.HTTP_201_CREATED)
async def create_entry(
    entry: JournalEntryCreate,
    db: Session = Depends(get_db),
):
    """Create a new journal entry with automatic analysis."""
    try:
        analysis_result = get_analyzer().analyze(entry.text)

        repo = JournalRepository(db)
        journal_entry = repo.create_entry(
            text=entry.text,
            user_id=entry.user_id,
            analysis_result=analysis_result,
        )

        return JournalEntryResponse(
            id=journal_entry.id,
            text=journal_entry.text,
            timestamp=journal_entry.timestamp,
            user_id=journal_entry.user_id,
            analysis=AnalysisResult(
                emotions=analysis_result.get("emotions"),
                distortions=analysis_result.get("distortions"),
                insights=analysis_result.get("insights", []),
                recommendations=analysis_result.get("recommendations", []),
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating entry: {str(e)}",
        )


@app.get("/api/entries", response_model=JournalEntryListResponse)
async def get_entries(
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """List journal entries with pagination and date filtering."""
    repo = JournalRepository(db)
    entries = repo.get_entries(
        user_id=user_id,
        skip=skip,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )

    entry_responses = [entry_to_response(entry) for entry in entries]
    total = repo.count_entries(user_id=user_id, start_date=start_date, end_date=end_date)

    return JournalEntryListResponse(
        entries=entry_responses,
        total=total,
        skip=skip,
        limit=limit,
    )


@app.get("/api/entries/trends", response_model=List[TrendDataPoint])
async def get_trends(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    emotion: Optional[str] = None,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """Get emotional trends over time."""
    repo = JournalRepository(db)
    trends = repo.get_emotion_trends(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        emotion=emotion,
    )
    return [TrendDataPoint(**trend) for trend in trends]


@app.get("/api/entries/stats", response_model=StatisticsResponse)
async def get_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """Get aggregated emotion statistics."""
    repo = JournalRepository(db)
    stats = repo.get_emotion_statistics(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    return StatisticsResponse(
        emotions=[EmotionStatistics(**emotion) for emotion in stats["emotions"]],
        total_entries=stats["total_entries"],
    )


@app.get("/api/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_entry(
    entry_id: int,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """Get a specific journal entry by ID."""
    repo = JournalRepository(db)
    entry = repo.get_entry(entry_id, user_id)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found",
        )

    return entry_to_response(entry)


@app.delete("/api/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entry(
    entry_id: int,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """Delete a journal entry."""
    repo = JournalRepository(db)
    deleted = repo.delete_entry(entry_id, user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found",
        )

    return None


# ---------------------------------------------------------------------------
# Distortion statistics
# ---------------------------------------------------------------------------
@app.get("/api/distortions/stats")
async def get_distortion_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """Get aggregated cognitive-distortion statistics."""
    repo = JournalRepository(db)
    return repo.get_distortion_statistics(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# Stateless analysis (no persistence)
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text without saving it to the database."""
    try:
        result = get_analyzer().analyze(request.text)
        return AnalysisResult(
            emotions=result.get("emotions"),
            distortions=result.get("distortions"),
            insights=result.get("insights", []),
            recommendations=result.get("recommendations", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )
