"""
Trend analysis and pattern detection for journal entries
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from app.repository import JournalRepository
from app.models import JournalEntry


class TrendAnalyzer:
    """
    Analyze emotional evolution and patterns in journal entries
    """
    
    def __init__(self, repository: JournalRepository):
        self.repo = repository
    
    def get_emotional_evolution(
        self,
        user_id: str = "default",
        days: int = 30,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get emotional evolution over time
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            emotion: Specific emotion to track (optional)
        
        Returns:
            Dictionary with evolution data and insights
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        trends = self.repo.get_emotion_trends(
            user_id=user_id,
            start_date=start_date,
            emotion=emotion
        )
        
        if not trends:
            return {
                'data': [],
                'insights': ["No data available for the selected period"],
                'patterns': []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(trends)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate changes
        insights = []
        patterns = []
        
        if emotion:
            # Single emotion tracking
            emotion_data = df[df['emotion'] == emotion].sort_values('date')
            if len(emotion_data) > 1:
                first_prob = emotion_data.iloc[0]['avg_probability']
                last_prob = emotion_data.iloc[-1]['avg_probability']
                change = last_prob - first_prob
                
                if abs(change) > 0.1:
                    direction = "increasing" if change > 0 else "decreasing"
                    insights.append(
                        f"{emotion.title()} shows {direction} trend "
                        f"({change:+.1%} change over period)"
                    )
        else:
            # Multi-emotion analysis
            emotion_changes = {}
            for emo in df['emotion'].unique():
                emo_data = df[df['emotion'] == emo].sort_values('date')
                if len(emo_data) > 1:
                    change = emo_data.iloc[-1]['avg_probability'] - emo_data.iloc[0]['avg_probability']
                    emotion_changes[emo] = change
            
            # Find most significant changes
            sorted_changes = sorted(emotion_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            for emo, change in sorted_changes[:3]:
                if abs(change) > 0.1:
                    direction = "increasing" if change > 0 else "decreasing"
                    insights.append(
                        f"{emo.title()} is {direction} ({change:+.1%})"
                    )
        
        # Detect patterns
        patterns = self._detect_patterns(df)
        
        return {
            'data': trends,
            'insights': insights if insights else ["Stable emotional patterns detected"],
            'patterns': patterns
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect patterns in emotional data
        
        Args:
            df: DataFrame with trend data
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Group by emotion and check for trends
        for emotion in df['emotion'].unique():
            emo_data = df[df['emotion'] == emotion].sort_values('date')
            
            if len(emo_data) < 3:
                continue
            
            # Check for consistent increase/decrease
            probs = emo_data['avg_probability'].values
            if len(probs) >= 3:
                # Simple trend detection
                increasing = all(probs[i] <= probs[i+1] for i in range(len(probs)-1))
                decreasing = all(probs[i] >= probs[i+1] for i in range(len(probs)-1))
                
                if increasing and (probs[-1] - probs[0]) > 0.15:
                    patterns.append(f"{emotion.title()} showing consistent increase")
                elif decreasing and (probs[0] - probs[-1]) > 0.15:
                    patterns.append(f"{emotion.title()} showing consistent decrease")
        
        return patterns
    
    def compare_periods(
        self,
        user_id: str = "default",
        period1_days: int = 7,
        period2_days: int = 7
    ) -> Dict[str, Any]:
        """
        Compare emotional patterns between two time periods
        
        Args:
            user_id: User identifier
            period1_days: Days for first period (most recent)
            period2_days: Days for second period (before first)
        
        Returns:
            Comparison analysis
        """
        now = datetime.utcnow()
        period1_start = now - timedelta(days=period1_days)
        period2_end = period1_start
        period2_start = period2_end - timedelta(days=period2_days)
        
        # Get statistics for both periods
        stats1 = self.repo.get_emotion_statistics(
            user_id=user_id,
            start_date=period1_start
        )
        stats2 = self.repo.get_emotion_statistics(
            user_id=user_id,
            start_date=period2_start,
            end_date=period2_end
        )
        
        # Create comparison
        comparison = {
            'period1': {
                'label': f'Last {period1_days} days',
                'stats': stats1
            },
            'period2': {
                'label': f'Previous {period2_days} days',
                'stats': stats2
            },
            'changes': []
        }
        
        # Calculate changes
        emotions1 = {e['emotion']: e for e in stats1['emotions']}
        emotions2 = {e['emotion']: e for e in stats2['emotions']}
        
        all_emotions = set(emotions1.keys()) | set(emotions2.keys())
        
        for emotion in all_emotions:
            e1 = emotions1.get(emotion, {'avg_probability': 0, 'count': 0})
            e2 = emotions2.get(emotion, {'avg_probability': 0, 'count': 0})
            
            prob_change = e1['avg_probability'] - e2['avg_probability']
            count_change = e1['count'] - e2['count']
            
            if abs(prob_change) > 0.05 or abs(count_change) > 0:
                comparison['changes'].append({
                    'emotion': emotion,
                    'probability_change': prob_change,
                    'count_change': count_change,
                    'direction': 'increasing' if prob_change > 0 else 'decreasing'
                })
        
        # Sort by magnitude of change
        comparison['changes'].sort(key=lambda x: abs(x['probability_change']), reverse=True)
        
        return comparison
    
    def get_weekly_summary(
        self,
        user_id: str = "default",
        weeks: int = 4
    ) -> Dict[str, Any]:
        """
        Get weekly emotional summaries
        
        Args:
            user_id: User identifier
            weeks: Number of weeks to analyze
        
        Returns:
            Weekly summary data
        """
        summaries = []
        now = datetime.utcnow()
        
        for week in range(weeks):
            week_start = now - timedelta(days=(week + 1) * 7)
            week_end = now - timedelta(days=week * 7)
            
            stats = self.repo.get_emotion_statistics(
                user_id=user_id,
                start_date=week_start,
                end_date=week_end
            )
            
            # Get top emotions
            top_emotions = sorted(
                stats['emotions'],
                key=lambda x: x['avg_probability'],
                reverse=True
            )[:3]
            
            summaries.append({
                'week': f"Week {weeks - week}",
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'total_entries': stats['total_entries'],
                'top_emotions': [e['emotion'] for e in top_emotions],
                'emotion_details': top_emotions
            })
        
        return {
            'summaries': summaries,
            'weeks_analyzed': weeks
        }
    
    def get_distortion_frequency(
        self,
        user_id: str = "default",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze frequency of cognitive distortions
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
        
        Returns:
            Distortion frequency analysis
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        stats = self.repo.get_distortion_statistics(
            user_id=user_id,
            start_date=start_date
        )
        
        insights = []
        
        if stats['distortions']:
            # Most common distortion
            most_common = max(stats['distortions'], key=lambda x: x['count'])
            insights.append(
                f"Most common distortion: {most_common['name']} "
                f"({most_common['count']} occurrences)"
            )
            
            # Total distortion count
            total_distortions = sum(d['count'] for d in stats['distortions'])
            total_entries = stats['total_entries_with_distortions']
            
            if total_entries > 0:
                avg_per_entry = total_distortions / total_entries
                insights.append(
                    f"Average {avg_per_entry:.1f} distortion(s) per entry with distortions"
                )
        
        return {
            'statistics': stats,
            'insights': insights
        }
    
    def detect_emotional_patterns(
        self,
        user_id: str = "default",
        days: int = 30
    ) -> List[str]:
        """
        Detect high-level emotional patterns
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
        
        Returns:
            List of detected patterns
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        stats = self.repo.get_emotion_statistics(
            user_id=user_id,
            start_date=start_date
        )
        
        patterns = []
        
        if not stats['emotions']:
            return ["No data available"]
        
        # Categorize emotions
        negative_emotions = ['sadness', 'anger', 'fear', 'anxiety', 'disappointment', 'grief', 'remorse']
        positive_emotions = ['joy', 'happiness', 'excitement', 'gratitude', 'love', 'pride', 'optimism']
        
        negative_count = sum(
            e['count'] for e in stats['emotions']
            if e['emotion'] in negative_emotions
        )
        positive_count = sum(
            e['count'] for e in stats['emotions']
            if e['emotion'] in positive_emotions
        )
        
        total_emotions = sum(e['count'] for e in stats['emotions'])
        
        if total_emotions > 0:
            negative_ratio = negative_count / total_emotions
            positive_ratio = positive_count / total_emotions
            
            if negative_ratio > 0.5:
                patterns.append("Higher frequency of negative emotions detected")
            elif positive_ratio > 0.5:
                patterns.append("Higher frequency of positive emotions detected")
            else:
                patterns.append("Balanced emotional patterns")
        
        # Check for specific emotion dominance
        if stats['emotions']:
            top_emotion = max(stats['emotions'], key=lambda x: x['count'])
            if top_emotion['count'] > total_emotions * 0.3:
                patterns.append(
                    f"{top_emotion['emotion'].title()} appears frequently "
                    f"({top_emotion['count']} occurrences)"
                )
        
        return patterns
