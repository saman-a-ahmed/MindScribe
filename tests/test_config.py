"""Tests for app.config.Settings."""

from app.config import settings


def test_core_settings_present_and_typed():
    assert settings.APP_NAME
    assert settings.APP_VERSION
    assert isinstance(settings.EMOTION_THRESHOLD, float)
    assert isinstance(settings.DISTORTION_THRESHOLD, float)
    assert 0.0 <= settings.EMOTION_THRESHOLD <= 1.0
    assert 0.0 <= settings.DISTORTION_THRESHOLD <= 1.0


def test_cors_origins_is_list_of_str():
    assert isinstance(settings.CORS_ORIGINS, list)
    assert settings.CORS_ORIGINS  # at least one origin
    assert all(isinstance(origin, str) for origin in settings.CORS_ORIGINS)


def test_validate_returns_true():
    assert settings.validate() is True
