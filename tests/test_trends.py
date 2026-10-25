"""Tests for app.trends.TrendAnalyzer (built on the repository)."""

from datetime import timedelta

from app.trends import TrendAnalyzer
from app.utils import utcnow


def test_emotional_evolution_no_data(repo):
    analyzer = TrendAnalyzer(repo)
    result = analyzer.get_emotional_evolution(days=30)
    assert result["data"] == []
    assert result["insights"] == ["No data available for the selected period"]
    assert result["patterns"] == []


def test_emotional_evolution_detects_increase(add_entry, repo):
    now = utcnow()
    add_entry(timestamp=now - timedelta(days=10), emotions=[("sadness", 0.3)])
    add_entry(timestamp=now - timedelta(days=2), emotions=[("sadness", 0.7)])

    analyzer = TrendAnalyzer(repo)
    result = analyzer.get_emotional_evolution(days=30)

    joined = " ".join(result["insights"]).lower()
    assert "sadness" in joined
    assert "increasing" in joined


def test_detect_emotional_patterns_negative(add_entry, repo):
    now = utcnow()
    for i in range(3):
        add_entry(timestamp=now - timedelta(days=i + 1), emotions=[("sadness", 0.6)])

    analyzer = TrendAnalyzer(repo)
    patterns = analyzer.detect_emotional_patterns(days=30)
    assert "Higher frequency of negative emotions detected" in patterns


def test_compare_periods_structure(add_entry, repo):
    now = utcnow()
    add_entry(timestamp=now - timedelta(days=3), emotions=[("joy", 0.8)])
    add_entry(timestamp=now - timedelta(days=10), emotions=[("sadness", 0.5)])

    analyzer = TrendAnalyzer(repo)
    comparison = analyzer.compare_periods(period1_days=7, period2_days=7)

    assert set(comparison) == {"period1", "period2", "changes"}
    assert isinstance(comparison["changes"], list)


def test_distortion_frequency_insights(add_entry, repo):
    now = utcnow()
    add_entry(
        timestamp=now - timedelta(days=1),
        distortions=[("catastrophizing", "Catastrophizing", "d", 0.8)],
    )
    add_entry(
        timestamp=now - timedelta(days=2),
        distortions=[("catastrophizing", "Catastrophizing", "d", 0.6)],
    )

    analyzer = TrendAnalyzer(repo)
    result = analyzer.get_distortion_frequency(days=30)

    assert result["insights"]
    assert any("Catastrophizing" in msg for msg in result["insights"])
