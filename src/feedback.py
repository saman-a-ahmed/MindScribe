"""
Model-free feedback generation for analysed journal entries.

These helpers turn an *analysis result* dictionary (the structure produced by
:meth:`src.analyzer.JournalAnalyzer.analyze`, or reconstructed from stored
database rows) into human-readable ``insights`` and ``recommendations``. They
deliberately depend on **no machine-learning libraries**, so they can run on the
API read path and be unit-tested without loading any model.

An *analysis result* looks like::

    {
        "emotions": {
            "top_emotions": [{"emotion": "sadness", "probability": 0.8}, ...],
            "detected": [...],
            "primary_emotion": {...},
        },
        "distortions": {
            "detected": [{"label": "catastrophizing", "name": ..., "probability": ...}, ...],
            ...
        },
    }
"""

from typing import Any, Dict, List

# Emotion taxonomy buckets shared across the analyzer and trend analysis.
# Centralised here so the API, the analyzer, and app.trends stay consistent.
NEGATIVE_EMOTIONS: List[str] = [
    "sadness", "anger", "fear", "anxiety", "disappointment", "grief", "remorse",
]
POSITIVE_EMOTIONS: List[str] = [
    "joy", "happiness", "excitement", "gratitude", "love", "pride", "optimism",
]

# Recommendation copy keyed by the cognitive-distortion label that triggers it.
DISTORTION_RECOMMENDATIONS: Dict[str, str] = {
    "catastrophizing": "Consider challenging catastrophic thoughts - what's the most likely outcome?",
    "all_or_nothing": "Try to find middle ground - situations are rarely all-or-nothing",
    "mind_reading": "Remember: you can't know what others are thinking without asking",
    "should_statements": "Replace 'should' statements with more flexible thinking",
    "personalization": "Consider: are you taking responsibility for things outside your control?",
    "overgeneralization": "One event doesn't define everything - look for exceptions",
}

# Recommendation copy keyed by the dominant (primary) emotion.
EMOTION_RECOMMENDATIONS: List[tuple] = [
    (("sadness", "grief", "disappointment"),
     "Consider self-compassion practices and acknowledging your feelings"),
    (("anxiety", "fear", "nervousness"),
     "Try grounding techniques or deep breathing exercises"),
    (("anger", "annoyance"),
     "Consider what's beneath the anger - what need isn't being met?"),
]


def _top_emotions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Safely pull the ``top_emotions`` list from an analysis result."""
    emotions = analysis.get("emotions")
    if isinstance(emotions, dict):
        return emotions.get("top_emotions") or []
    return []


def _detected_distortions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Safely pull the detected-distortions list from an analysis result."""
    distortions = analysis.get("distortions")
    if isinstance(distortions, dict) and "detected" in distortions:
        return distortions.get("detected") or []
    return []


def _has_distortion_key(analysis: Dict[str, Any]) -> bool:
    """Whether distortion analysis ran (the ``detected`` key is present)."""
    distortions = analysis.get("distortions")
    return isinstance(distortions, dict) and "detected" in distortions


def derive_insights(analysis: Dict[str, Any]) -> List[str]:
    """
    Generate insight strings from an analysis result.

    Args:
        analysis: Analysis result dictionary (see module docstring).

    Returns:
        Ordered list of insight strings (may be empty).
    """
    insights: List[str] = []

    # Emotion insights
    top_emotions = _top_emotions(analysis)
    if top_emotions:
        primary = top_emotions[0]
        insights.append(
            f"Primary emotion detected: {primary['emotion']} "
            f"({primary['probability']:.1%} confidence)"
        )
        detected_negative = [e for e in top_emotions if e["emotion"] in NEGATIVE_EMOTIONS]
        if detected_negative:
            names = ", ".join(e["emotion"] for e in detected_negative)
            insights.append(f"Negative emotions present: {names}")

    # Distortion insights
    if _has_distortion_key(analysis):
        detected = _detected_distortions(analysis)
        if detected:
            insights.append(f"Detected {len(detected)} cognitive distortion(s)")
            if len(detected) > 2:
                insights.append(
                    "Multiple cognitive distortions detected - "
                    "consider reflecting on thought patterns"
                )
        else:
            insights.append(
                "No significant cognitive distortions detected - balanced thinking patterns"
            )

    # Combined insight
    if top_emotions and _detected_distortions(analysis):
        insights.append(
            "Emotional patterns and thought distortions identified - "
            "consider exploring connections"
        )

    return insights


def derive_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """
    Generate recommendation strings from an analysis result.

    Args:
        analysis: Analysis result dictionary (see module docstring).

    Returns:
        Ordered list of recommendation strings; never empty (falls back to a
        generic journaling suggestion).
    """
    recommendations: List[str] = []

    # Distortion-based recommendations
    detected = _detected_distortions(analysis)
    if detected:
        distortion_types = [d["label"] for d in detected]
        for label, tip in DISTORTION_RECOMMENDATIONS.items():
            if label in distortion_types:
                recommendations.append(tip)
        recommendations.append("Consider practicing cognitive restructuring techniques from CBT")

    # Emotion-based recommendations (driven by the primary emotion)
    top_emotions = _top_emotions(analysis)
    if top_emotions:
        primary = top_emotions[0]["emotion"]
        for emotions, tip in EMOTION_RECOMMENDATIONS:
            if primary in emotions:
                recommendations.append(tip)

    if not recommendations:
        recommendations.append("Continue journaling regularly to track patterns over time")

    return recommendations
