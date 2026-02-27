"""
Unified analysis service combining emotion classification and cognitive distortion detection
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Handle imports - add src directory to path if needed
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from same directory
from inference import EmotionClassifier, EMOTION_LABELS
from inference_distortion import DistortionClassifier


class JournalAnalyzer:
    """
    Unified service for analyzing journal entries using both emotion and distortion models
    """
    
    def __init__(
        self,
        emotion_model_path: str = "models/emotion_classifier",
        distortion_model_path: str = "models/distortion_classifier",
        emotion_threshold: float = 0.3,
        distortion_threshold: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize the analyzer with both models
        
        Args:
            emotion_model_path: Path to emotion classification model
            distortion_model_path: Path to distortion classification model
            emotion_threshold: Probability threshold for emotion detection
            distortion_threshold: Probability threshold for distortion detection
            verbose: Whether to print loading messages
        """
        self.emotion_threshold = emotion_threshold
        self.distortion_threshold = distortion_threshold
        self.verbose = verbose
        
        # Load emotion classifier
        self.emotion_classifier = None
        if os.path.exists(emotion_model_path):
            try:
                if verbose:
                    print(f"Loading emotion classifier from {emotion_model_path}...")
                self.emotion_classifier = EmotionClassifier(emotion_model_path)
                if verbose:
                    print("✓ Emotion classifier loaded")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to load emotion classifier: {e}")
        else:
            if verbose:
                print(f"⚠️  Emotion model not found at {emotion_model_path}")
        
        # Load distortion classifier
        self.distortion_classifier = None
        if os.path.exists(distortion_model_path):
            try:
                if verbose:
                    print(f"Loading distortion classifier from {distortion_model_path}...")
                self.distortion_classifier = DistortionClassifier(distortion_model_path)
                if verbose:
                    print("✓ Distortion classifier loaded")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to load distortion classifier: {e}")
        else:
            if verbose:
                print(f"⚠️  Distortion model not found at {distortion_model_path}")
        
        if not self.emotion_classifier and not self.distortion_classifier:
            raise RuntimeError("Neither emotion nor distortion classifier could be loaded")
    
    def analyze(
        self,
        text: str,
        emotion_threshold: Optional[float] = None,
        distortion_threshold: Optional[float] = None,
        top_k_emotions: int = 5,
        top_k_distortions: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze a journal entry for both emotions and cognitive distortions
        
        Args:
            text: Journal entry text to analyze
            emotion_threshold: Override default emotion threshold
            distortion_threshold: Override default distortion threshold
            top_k_emotions: Number of top emotions to return
            top_k_distortions: Number of top distortions to return
        
        Returns:
            Dictionary with complete analysis results
        """
        if not text or not text.strip():
            return {
                'text': text,
                'error': 'Empty text provided',
                'emotions': None,
                'distortions': None
            }
        
        emotion_threshold = emotion_threshold or self.emotion_threshold
        distortion_threshold = distortion_threshold or self.distortion_threshold
        
        result = {
            'text': text,
            'emotions': None,
            'distortions': None,
            'insights': [],
            'recommendations': []
        }
        
        # Analyze emotions
        if self.emotion_classifier:
            try:
                emotion_result = self.emotion_classifier.predict(
                    text,
                    threshold=emotion_threshold,
                    top_k=top_k_emotions
                )
                result['emotions'] = {
                    'detected': emotion_result['all_emotions'],
                    'top_emotions': emotion_result['top_emotions'],
                    'primary_emotion': emotion_result['top_emotions'][0] if emotion_result['top_emotions'] else None
                }
            except Exception as e:
                result['emotions'] = {'error': str(e)}
        else:
            result['emotions'] = {'error': 'Emotion classifier not available'}
        
        # Analyze distortions
        if self.distortion_classifier:
            try:
                distortion_result = self.distortion_classifier.predict(
                    text,
                    threshold=distortion_threshold,
                    top_k=top_k_distortions
                )
                result['distortions'] = {
                    'detected': distortion_result['detected_distortions'],
                    'top_distortions': distortion_result['top_distortions'],
                    'has_distortions': distortion_result['has_distortions'],
                    'count': len(distortion_result['detected_distortions'])
                }
            except Exception as e:
                result['distortions'] = {'error': str(e)}
        else:
            result['distortions'] = {'error': 'Distortion classifier not available'}
        
        # Generate insights
        result['insights'] = self._generate_insights(result)
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
    
    def _generate_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate insights based on analysis results
        
        Args:
            analysis_result: Result from analyze() method
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # Emotion insights
        if analysis_result.get('emotions') and 'top_emotions' in analysis_result['emotions']:
            top_emotions = analysis_result['emotions']['top_emotions']
            if top_emotions:
                primary = top_emotions[0]
                insights.append(f"Primary emotion detected: {primary['emotion']} ({primary['probability']:.1%} confidence)")
                
                # Check for negative emotions
                negative_emotions = ['sadness', 'anger', 'fear', 'anxiety', 'disappointment', 'grief']
                detected_negative = [e for e in top_emotions if e['emotion'] in negative_emotions]
                if detected_negative:
                    insights.append(f"Negative emotions present: {', '.join([e['emotion'] for e in detected_negative])}")
        
        # Distortion insights
        if analysis_result.get('distortions') and 'detected' in analysis_result['distortions']:
            detected = analysis_result['distortions']['detected']
            if detected:
                insights.append(f"Detected {len(detected)} cognitive distortion(s)")
                if len(detected) > 2:
                    insights.append("Multiple cognitive distortions detected - consider reflecting on thought patterns")
            else:
                insights.append("No significant cognitive distortions detected - balanced thinking patterns")
        
        # Combined insights
        if (analysis_result.get('emotions') and analysis_result.get('distortions') and
            'detected' in analysis_result['distortions'] and
            analysis_result['distortions']['detected']):
            insights.append("Emotional patterns and thought distortions identified - consider exploring connections")
        
        return insights
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis results
        
        Args:
            analysis_result: Result from analyze() method
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Distortion-based recommendations
        if analysis_result.get('distortions') and 'detected' in analysis_result['distortions']:
            detected = analysis_result['distortions']['detected']
            if detected:
                # Get most common distortion types
                distortion_types = [d['label'] for d in detected]
                
                if 'catastrophizing' in distortion_types:
                    recommendations.append("Consider challenging catastrophic thoughts - what's the most likely outcome?")
                
                if 'all_or_nothing' in distortion_types:
                    recommendations.append("Try to find middle ground - situations are rarely all-or-nothing")
                
                if 'mind_reading' in distortion_types:
                    recommendations.append("Remember: you can't know what others are thinking without asking")
                
                if 'should_statements' in distortion_types:
                    recommendations.append("Replace 'should' statements with more flexible thinking")
                
                if 'personalization' in distortion_types:
                    recommendations.append("Consider: are you taking responsibility for things outside your control?")
                
                if 'overgeneralization' in distortion_types:
                    recommendations.append("One event doesn't define everything - look for exceptions")
                
                recommendations.append("Consider practicing cognitive restructuring techniques from CBT")
        
        # Emotion-based recommendations
        if analysis_result.get('emotions') and 'top_emotions' in analysis_result['emotions']:
            top_emotions = analysis_result['emotions']['top_emotions']
            if top_emotions:
                primary = top_emotions[0]['emotion']
                
                if primary in ['sadness', 'grief', 'disappointment']:
                    recommendations.append("Consider self-compassion practices and acknowledging your feelings")
                
                if primary in ['anxiety', 'fear', 'nervousness']:
                    recommendations.append("Try grounding techniques or deep breathing exercises")
                
                if primary in ['anger', 'annoyance']:
                    recommendations.append("Consider what's beneath the anger - what need isn't being met?")
        
        if not recommendations:
            recommendations.append("Continue journaling regularly to track patterns over time")
        
        return recommendations
    
    def analyze_batch(
        self,
        texts: List[str],
        emotion_threshold: Optional[float] = None,
        distortion_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple journal entries
        
        Args:
            texts: List of journal entry texts
            emotion_threshold: Override default emotion threshold
            distortion_threshold: Override default distortion threshold
        
        Returns:
            List of analysis results
        """
        return [self.analyze(text, emotion_threshold, distortion_threshold) for text in texts]
    
    def is_ready(self) -> bool:
        """
        Check if analyzer is ready (at least one model loaded)
        
        Returns:
            True if at least one classifier is available
        """
        return self.emotion_classifier is not None or self.distortion_classifier is not None
    
    def get_status(self) -> Dict[str, bool]:
        """
        Get status of loaded models
        
        Returns:
            Dictionary with model availability status
        """
        return {
            'emotion_classifier': self.emotion_classifier is not None,
            'distortion_classifier': self.distortion_classifier is not None,
            'ready': self.is_ready()
        }
