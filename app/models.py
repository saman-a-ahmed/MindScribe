"""
SQLAlchemy database models for MindScribe
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.database import Base
from app.utils import utcnow


class JournalEntry(Base):
    """
    Journal entry model - stores user journal entries
    """
    __tablename__ = "journal_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False, index=True)
    user_id = Column(String(100), default="default", nullable=False, index=True)  # For future multi-user support
    
    # Relationships
    emotions = relationship("EmotionAnalysis", back_populates="entry", cascade="all, delete-orphan")
    distortions = relationship("DistortionAnalysis", back_populates="entry", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<JournalEntry(id={self.id}, timestamp={self.timestamp}, text_length={len(self.text)})>"


class EmotionAnalysis(Base):
    """
    Emotion analysis results for journal entries
    """
    __tablename__ = "emotion_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, ForeignKey("journal_entries.id"), nullable=False, index=True)
    emotion = Column(String(50), nullable=False, index=True)
    probability = Column(Float, nullable=False)
    
    # Relationship
    entry = relationship("JournalEntry", back_populates="emotions")
    
    def __repr__(self):
        return f"<EmotionAnalysis(entry_id={self.entry_id}, emotion={self.emotion}, prob={self.probability:.3f})>"


class DistortionAnalysis(Base):
    """
    Cognitive distortion analysis results for journal entries
    """
    __tablename__ = "distortion_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, ForeignKey("journal_entries.id"), nullable=False, index=True)
    distortion_type = Column(String(50), nullable=False, index=True)  # e.g., 'all_or_nothing'
    distortion_name = Column(String(100), nullable=False)  # Human-readable name
    description = Column(Text)  # Description of the distortion
    probability = Column(Float, nullable=False)
    
    # Relationship
    entry = relationship("JournalEntry", back_populates="distortions")
    
    def __repr__(self):
        return f"<DistortionAnalysis(entry_id={self.entry_id}, type={self.distortion_type}, prob={self.probability:.3f})>"
