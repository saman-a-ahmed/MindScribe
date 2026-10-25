"""Tests for src.feedback (model-free insights/recommendations)."""

from src.feedback import derive_insights, derive_recommendations


def _analysis(top_emotions=None, detected_distortions=None, has_distortion_key=True):
    distortions = None
    if has_distortion_key:
        distortions = {"detected": detected_distortions or []}
    return {
        "emotions": {"top_emotions": top_emotions or []},
        "distortions": distortions,
    }


def test_insights_primary_and_negative():
    analysis = _analysis(
        top_emotions=[
            {"emotion": "sadness", "probability": 0.8},
            {"emotion": "grief", "probability": 0.4},
        ],
        detected_distortions=[],
    )
    insights = derive_insights(analysis)
    assert insights[0].startswith("Primary emotion detected: sadness")
    assert any("Negative emotions present" in i for i in insights)


def test_insights_no_distortions_message():
    analysis = _analysis(
        top_emotions=[{"emotion": "joy", "probability": 0.9}],
        detected_distortions=[],
    )
    insights = derive_insights(analysis)
    assert any("No significant cognitive distortions detected" in i for i in insights)


def test_insights_multiple_distortions():
    detected = [
        {"label": "catastrophizing", "probability": 0.8},
        {"label": "labeling", "probability": 0.7},
        {"label": "mind_reading", "probability": 0.6},
    ]
    analysis = _analysis(
        top_emotions=[{"emotion": "fear", "probability": 0.7}],
        detected_distortions=detected,
    )
    insights = derive_insights(analysis)
    assert any("Detected 3 cognitive distortion(s)" in i for i in insights)
    assert any("Multiple cognitive distortions" in i for i in insights)
    assert any("exploring connections" in i for i in insights)


def test_recommendations_distortion_specific():
    analysis = _analysis(
        top_emotions=[],
        detected_distortions=[{"label": "catastrophizing", "probability": 0.8}],
    )
    recs = derive_recommendations(analysis)
    assert any("catastrophic thoughts" in r for r in recs)
    assert any("cognitive restructuring" in r for r in recs)


def test_recommendations_emotion_primary():
    analysis = _analysis(
        top_emotions=[{"emotion": "anxiety", "probability": 0.7}],
        detected_distortions=[],
    )
    recs = derive_recommendations(analysis)
    assert any("grounding techniques" in r for r in recs)


def test_recommendations_default_when_empty():
    analysis = _analysis(top_emotions=[], detected_distortions=[])
    recs = derive_recommendations(analysis)
    assert recs == ["Continue journaling regularly to track patterns over time"]


def test_handles_missing_keys_gracefully():
    # No emotions/distortions keys at all
    assert derive_insights({}) == []
    assert derive_recommendations({}) == [
        "Continue journaling regularly to track patterns over time"
    ]
