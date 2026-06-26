"""
Data access layer - CRUD operations and queries for journal entries
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from app.models import JournalEntry, EmotionAnalysis, DistortionAnalysis
from app.utils import utcnow


class JournalRepository:
    """
    Repository for journal entry operations
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_entry(
        self,
        text: str,
        user_id: str = "default",
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> JournalEntry:
        """
        Create a new journal entry with optional analysis results
        
        Args:
            text: Journal entry text
            user_id: User identifier (default for single-user)
            analysis_result: Analysis result from JournalAnalyzer
        
        Returns:
            Created JournalEntry object
        """
        entry = JournalEntry(
            text=text,
            user_id=user_id,
            timestamp=utcnow()
        )
        self.db.add(entry)
        self.db.flush()  # Get the ID
        
        # Add emotion analyses if provided
        if analysis_result and analysis_result.get('emotions'):
            emotions = analysis_result['emotions']
            if 'detected' in emotions and emotions['detected']:
                for emotion_data in emotions['detected']:
                    emotion_analysis = EmotionAnalysis(
                        entry_id=entry.id,
                        emotion=emotion_data['emotion'],
                        probability=emotion_data['probability']
                    )
                    self.db.add(emotion_analysis)
        
        # Add distortion analyses if provided
        if analysis_result and analysis_result.get('distortions'):
            distortions = analysis_result['distortions']
            if 'detected' in distortions and distortions['detected']:
                for distortion_data in distortions['detected']:
                    distortion_analysis = DistortionAnalysis(
                        entry_id=entry.id,
                        distortion_type=distortion_data['label'],
                        distortion_name=distortion_data['name'],
                        description=distortion_data.get('description', ''),
                        probability=distortion_data['probability']
                    )
                    self.db.add(distortion_analysis)
        
        self.db.commit()
        self.db.refresh(entry)
        return entry
    
    def get_entry(self, entry_id: int, user_id: str = "default") -> Optional[JournalEntry]:
        """
        Get a journal entry by ID
        
        Args:
            entry_id: Entry ID
            user_id: User identifier
        
        Returns:
            JournalEntry or None if not found
        """
        return self.db.query(JournalEntry).filter(
            and_(
                JournalEntry.id == entry_id,
                JournalEntry.user_id == user_id
            )
        ).first()
    
    def get_entries(
        self,
        user_id: str = "default",
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[JournalEntry]:
        """
        Get journal entries with pagination and date filtering
        
        Args:
            user_id: User identifier
            skip: Number of entries to skip
            limit: Maximum number of entries to return
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of JournalEntry objects
        """
        query = self.db.query(JournalEntry).filter(
            JournalEntry.user_id == user_id
        )
        
        if start_date:
            query = query.filter(JournalEntry.timestamp >= start_date)
        if end_date:
            query = query.filter(JournalEntry.timestamp <= end_date)
        
        return query.order_by(desc(JournalEntry.timestamp)).offset(skip).limit(limit).all()

    def count_entries(
        self,
        user_id: str = "default",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Count journal entries matching the same filters as get_entries.

        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Total number of matching entries
        """
        query = self.db.query(func.count(JournalEntry.id)).filter(
            JournalEntry.user_id == user_id
        )

        if start_date:
            query = query.filter(JournalEntry.timestamp >= start_date)
        if end_date:
            query = query.filter(JournalEntry.timestamp <= end_date)

        return query.scalar() or 0

    def update_entry(self, entry_id: int, text: str, user_id: str = "default") -> Optional[JournalEntry]:
        """
        Update a journal entry's text
        
        Args:
            entry_id: Entry ID
            text: New text
            user_id: User identifier
        
        Returns:
            Updated JournalEntry or None if not found
        """
        entry = self.get_entry(entry_id, user_id)
        if entry:
            entry.text = text
            self.db.commit()
            self.db.refresh(entry)
        return entry
    
    def delete_entry(self, entry_id: int, user_id: str = "default") -> bool:
        """
        Delete a journal entry
        
        Args:
            entry_id: Entry ID
            user_id: User identifier
        
        Returns:
            True if deleted, False if not found
        """
        entry = self.get_entry(entry_id, user_id)
        if entry:
            self.db.delete(entry)
            self.db.commit()
            return True
        return False
    
    def get_emotion_trends(
        self,
        user_id: str = "default",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        emotion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get emotion trends over time
        
        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter
            emotion: Specific emotion to filter (optional)
        
        Returns:
            List of dictionaries with date, emotion, and average probability
        """
        query = self.db.query(
            func.date(JournalEntry.timestamp).label('date'),
            EmotionAnalysis.emotion,
            func.avg(EmotionAnalysis.probability).label('avg_probability'),
            func.count(EmotionAnalysis.id).label('count')
        ).join(
            EmotionAnalysis, JournalEntry.id == EmotionAnalysis.entry_id
        ).filter(
            JournalEntry.user_id == user_id
        )
        
        if start_date:
            query = query.filter(JournalEntry.timestamp >= start_date)
        if end_date:
            query = query.filter(JournalEntry.timestamp <= end_date)
        if emotion:
            query = query.filter(EmotionAnalysis.emotion == emotion)
        
        results = query.group_by(
            func.date(JournalEntry.timestamp),
            EmotionAnalysis.emotion
        ).order_by(
            func.date(JournalEntry.timestamp),
            EmotionAnalysis.emotion
        ).all()
        
        return [
            {
                'date': str(row.date),
                'emotion': row.emotion,
                'avg_probability': float(row.avg_probability),
                'count': row.count
            }
            for row in results
        ]
    
    def get_emotion_statistics(
        self,
        user_id: str = "default",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated emotion statistics
        
        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            Dictionary with emotion statistics
        """
        query = self.db.query(
            EmotionAnalysis.emotion,
            func.avg(EmotionAnalysis.probability).label('avg_probability'),
            func.count(EmotionAnalysis.id).label('count'),
            func.max(EmotionAnalysis.probability).label('max_probability')
        ).join(
            JournalEntry, EmotionAnalysis.entry_id == JournalEntry.id
        ).filter(
            JournalEntry.user_id == user_id
        )
        
        if start_date:
            query = query.filter(JournalEntry.timestamp >= start_date)
        if end_date:
            query = query.filter(JournalEntry.timestamp <= end_date)
        
        results = query.group_by(EmotionAnalysis.emotion).all()
        
        return {
            'emotions': [
                {
                    'emotion': row.emotion,
                    'avg_probability': float(row.avg_probability),
                    'count': row.count,
                    'max_probability': float(row.max_probability)
                }
                for row in results
            ],
            'total_entries': self.db.query(JournalEntry).filter(
                JournalEntry.user_id == user_id
            ).count()
        }
    
    def get_distortion_statistics(
        self,
        user_id: str = "default",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated distortion statistics
        
        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            Dictionary with distortion statistics
        """
        query = self.db.query(
            DistortionAnalysis.distortion_type,
            DistortionAnalysis.distortion_name,
            func.avg(DistortionAnalysis.probability).label('avg_probability'),
            func.count(DistortionAnalysis.id).label('count')
        ).join(
            JournalEntry, DistortionAnalysis.entry_id == JournalEntry.id
        ).filter(
            JournalEntry.user_id == user_id
        )
        
        if start_date:
            query = query.filter(JournalEntry.timestamp >= start_date)
        if end_date:
            query = query.filter(JournalEntry.timestamp <= end_date)
        
        results = query.group_by(
            DistortionAnalysis.distortion_type,
            DistortionAnalysis.distortion_name
        ).order_by(desc('count')).all()
        
        return {
            'distortions': [
                {
                    'type': row.distortion_type,
                    'name': row.distortion_name,
                    'avg_probability': float(row.avg_probability),
                    'count': row.count
                }
                for row in results
            ],
            'total_entries_with_distortions': self.db.query(
                func.count(func.distinct(DistortionAnalysis.entry_id))
            ).join(
                JournalEntry, DistortionAnalysis.entry_id == JournalEntry.id
            ).filter(
                JournalEntry.user_id == user_id
            ).scalar() or 0
        }
    
    def get_entries_with_distortions(
        self,
        user_id: str = "default",
        distortion_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[JournalEntry]:
        """
        Get entries that have cognitive distortions
        
        Args:
            user_id: User identifier
            distortion_type: Specific distortion type to filter (optional)
            skip: Number of entries to skip
            limit: Maximum number of entries to return
        
        Returns:
            List of JournalEntry objects
        """
        query = self.db.query(JournalEntry).join(
            DistortionAnalysis, JournalEntry.id == DistortionAnalysis.entry_id
        ).filter(
            JournalEntry.user_id == user_id
        )
        
        if distortion_type:
            query = query.filter(DistortionAnalysis.distortion_type == distortion_type)
        
        return query.distinct().order_by(desc(JournalEntry.timestamp)).offset(skip).limit(limit).all()
    
    def get_recent_entries(self, user_id: str = "default", days: int = 7) -> List[JournalEntry]:
        """
        Get entries from the last N days
        
        Args:
            user_id: User identifier
            days: Number of days to look back
        
        Returns:
            List of JournalEntry objects
        """
        start_date = utcnow() - timedelta(days=days)
        return self.get_entries(user_id=user_id, start_date=start_date)
