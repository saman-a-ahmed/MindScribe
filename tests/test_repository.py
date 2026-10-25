"""Tests for app.repository.JournalRepository (CRUD + aggregation queries)."""

from datetime import timedelta

from app.models import EmotionAnalysis, DistortionAnalysis
from app.utils import utcnow


def test_create_entry_persists_emotions_and_distortions(repo, db_session, make_analysis):
    analysis = make_analysis(
        emotions=[("joy", 0.9), ("optimism", 0.4)],
        distortions=[("catastrophizing", "Catastrophizing", "worst case", 0.8)],
    )
    entry = repo.create_entry(text="Big day!", analysis_result=analysis)

    assert entry.id is not None
    assert db_session.query(EmotionAnalysis).filter_by(entry_id=entry.id).count() == 2
    assert db_session.query(DistortionAnalysis).filter_by(entry_id=entry.id).count() == 1


def test_get_entry_and_missing(repo, make_analysis):
    entry = repo.create_entry(text="hello", analysis_result=make_analysis())
    assert repo.get_entry(entry.id).id == entry.id
    assert repo.get_entry(99999) is None
    # user scoping
    assert repo.get_entry(entry.id, user_id="someone_else") is None


def test_get_entries_pagination_and_order(add_entry, repo):
    now = utcnow()
    for i in range(5):
        add_entry(text=f"e{i}", timestamp=now - timedelta(days=i))

    entries = repo.get_entries(limit=3)
    assert len(entries) == 3
    # newest first
    assert entries[0].text == "e0"
    assert entries[1].text == "e1"

    page2 = repo.get_entries(skip=3, limit=3)
    assert len(page2) == 2
    assert page2[0].text == "e3"


def test_get_entries_date_filter(add_entry, repo):
    now = utcnow()
    add_entry(text="recent", timestamp=now - timedelta(days=1))
    add_entry(text="old", timestamp=now - timedelta(days=40))

    recent = repo.get_entries(start_date=now - timedelta(days=7))
    assert [e.text for e in recent] == ["recent"]


def test_count_entries(add_entry, repo):
    now = utcnow()
    add_entry(timestamp=now - timedelta(days=1))
    add_entry(timestamp=now - timedelta(days=2))
    add_entry(timestamp=now - timedelta(days=40))

    assert repo.count_entries() == 3
    assert repo.count_entries(start_date=now - timedelta(days=7)) == 2
    assert repo.count_entries(user_id="nobody") == 0


def test_delete_entry(repo, make_analysis):
    entry = repo.create_entry(text="bye", analysis_result=make_analysis())
    assert repo.delete_entry(entry.id) is True
    assert repo.get_entry(entry.id) is None
    assert repo.delete_entry(entry.id) is False


def test_get_emotion_trends(add_entry, repo):
    now = utcnow()
    add_entry(timestamp=now - timedelta(days=2), emotions=[("joy", 0.6)])
    add_entry(timestamp=now - timedelta(days=1), emotions=[("joy", 0.8)])

    trends = repo.get_emotion_trends(start_date=now - timedelta(days=30))
    assert trends
    joy_points = [t for t in trends if t["emotion"] == "joy"]
    assert len(joy_points) == 2
    assert all(set(t) == {"date", "emotion", "avg_probability", "count"} for t in trends)


def test_get_emotion_statistics(add_entry, repo):
    add_entry(emotions=[("joy", 0.6)])
    add_entry(emotions=[("joy", 0.8), ("sadness", 0.5)])

    stats = repo.get_emotion_statistics()
    assert stats["total_entries"] == 2
    joy = next(e for e in stats["emotions"] if e["emotion"] == "joy")
    assert joy["count"] == 2
    assert joy["avg_probability"] == 0.7
    assert joy["max_probability"] == 0.8


def test_get_distortion_statistics(add_entry, repo):
    add_entry(distortions=[("catastrophizing", "Catastrophizing", "d", 0.8)])
    add_entry(distortions=[("catastrophizing", "Catastrophizing", "d", 0.6)])
    add_entry(distortions=[("labeling", "Labeling", "d", 0.7)])

    stats = repo.get_distortion_statistics()
    assert stats["total_entries_with_distortions"] == 3
    cat = next(d for d in stats["distortions"] if d["type"] == "catastrophizing")
    assert cat["count"] == 2
    # ordered by count desc
    assert stats["distortions"][0]["count"] >= stats["distortions"][-1]["count"]


def test_get_recent_entries(add_entry, repo):
    now = utcnow()
    add_entry(text="recent", timestamp=now - timedelta(days=2))
    add_entry(text="old", timestamp=now - timedelta(days=20))

    recent = repo.get_recent_entries(days=7)
    assert [e.text for e in recent] == ["recent"]
