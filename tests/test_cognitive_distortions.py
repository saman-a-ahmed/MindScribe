"""Tests for the cognitive-distortion taxonomy (src.cognitive_distortions)."""

from src.cognitive_distortions import (
    COGNITIVE_DISTORTION_LABELS,
    DISTORTION_NAMES,
    DISTORTION_DESCRIPTIONS,
    SIGNAL_WORDS,
    EXAMPLE_TEXTS,
    get_num_distortions,
    get_all_distortions,
    get_distortion_info,
)


def test_num_distortions():
    assert get_num_distortions() == 12
    assert len(COGNITIVE_DISTORTION_LABELS) == 12


def test_taxonomy_keys_aligned():
    labels = set(COGNITIVE_DISTORTION_LABELS)
    assert set(DISTORTION_NAMES) == labels
    assert set(DISTORTION_DESCRIPTIONS) == labels
    assert set(SIGNAL_WORDS) == labels
    assert set(EXAMPLE_TEXTS) == labels


def test_get_all_distortions_returns_copy():
    all_labels = get_all_distortions()
    assert all_labels == COGNITIVE_DISTORTION_LABELS
    all_labels.append("tampered")
    assert "tampered" not in COGNITIVE_DISTORTION_LABELS


def test_get_distortion_info_by_index():
    info = get_distortion_info(0)
    assert info["label"] == COGNITIVE_DISTORTION_LABELS[0]
    assert info["name"] == DISTORTION_NAMES[info["label"]]
    assert "signal_words" in info and "examples" in info


def test_get_distortion_info_by_label():
    info = get_distortion_info("catastrophizing")
    assert info["name"] == "Catastrophizing"


def test_get_distortion_info_invalid():
    assert get_distortion_info(999) is None
    assert get_distortion_info("not_a_distortion") is None
    assert get_distortion_info(3.5) is None
