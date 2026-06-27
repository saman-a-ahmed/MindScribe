"""
Unified analysis service combining emotion classification and cognitive distortion detection
"""

import os
from typing import Any, Dict, List, Optional

from src.inference import EmotionClassifier
from src.inference_distortion import DistortionClassifier
from src.feedback import derive_insights, derive_recommendations


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
        
        emotion_threshold = (
            emotion_threshold if emotion_threshold is not None else self.emotion_threshold
        )
        distortion_threshold = (
            distortion_threshold if distortion_threshold is not None else self.distortion_threshold
        )
        
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
        
        # Generate insights and recommendations (model-free; see src.feedback)
        result['insights'] = derive_insights(result)
        result['recommendations'] = derive_recommendations(result)

        return result

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
