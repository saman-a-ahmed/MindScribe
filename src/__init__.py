"""
MindScribe machine-learning package.

Houses the emotion and cognitive-distortion classifiers, the unified
:class:`~src.analyzer.JournalAnalyzer`, model-free feedback generation
(:mod:`src.feedback`), and the training / preprocessing pipelines.

Modules import each other with absolute paths (``from src.<module> import ...``),
so the command-line entry points are designed to be run as modules from the
project root, e.g. ``python -m src.inference``.
"""
