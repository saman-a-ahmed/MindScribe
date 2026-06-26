# MindScribe Architecture

This document explains how MindScribe is put together: the data flow, the
responsibility of each module, and the import/run conventions the codebase
follows.

## High-level data flow

```
                    ┌──────────────────────────┐
   journal text ──▶ │  JournalAnalyzer          │  (src/analyzer.py)
                    │   ├─ EmotionClassifier     │  (src/inference.py)
                    │   └─ DistortionClassifier  │  (src/inference_distortion.py)
                    └──────────┬───────────────┘
                               │ analysis result (dict)
                               ▼
              derive_insights / derive_recommendations   (src/feedback.py)
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  JournalRepository        │  (app/repository.py)
                    │  persists entry +         │
                    │  emotions + distortions   │
                    └──────────┬───────────────┘
                               │ SQLAlchemy
                               ▼
                       Database (SQLite / PostgreSQL)
                               │
            ┌──────────────────┴───────────────────┐
            ▼                                       ▼
   FastAPI API (app/main.py)              Streamlit UI (app/streamlit_app.py)
   reads rebuild analysis from            TrendAnalyzer (app/trends.py)
   stored rows — no re-inference          surfaces insights & patterns
```

Key idea: **machine-learning inference runs only when text is *created* or
*analyzed***. Read endpoints (`GET /api/entries`, `GET /api/entries/{id}`)
rebuild their emotion/distortion payload from rows already persisted in the
database and regenerate insights/recommendations with the model-free helpers in
`src/feedback.py`. This keeps reads fast and independent of model availability.

## Modules

### `src/` — machine learning
| Module | Responsibility |
| --- | --- |
| `cognitive_distortions.py` | Taxonomy of the 12 CBT distortions: labels, names, descriptions, signal words, example texts. Single source of truth for distortion metadata. |
| `inference.py` | `EmotionClassifier` — loads the RoBERTa emotion model and predicts 28 GoEmotions labels. Also a CLI for sanity checks. |
| `inference_distortion.py` | `DistortionClassifier` — loads the RoBERTa distortion model and predicts the 12 distortion labels. Also a CLI. |
| `analyzer.py` | `JournalAnalyzer` — wraps both classifiers, returns a unified analysis dict, and attaches insights/recommendations. |
| `feedback.py` | Model-free `derive_insights` / `derive_recommendations` plus the shared `NEGATIVE_EMOTIONS` / `POSITIVE_EMOTIONS` buckets. No ML imports — safe to use anywhere. |
| `preprocessing.py`, `train_emotion.py` | GoEmotions preprocessing + emotion model training. |
| `generate_distortion_data.py`, `preprocess_distortions.py`, `train_distortion.py` | Synthetic distortion dataset generation, preprocessing, and training. |

### `app/` — web application
| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` loaded from environment / `.env`. Single source of truth for configuration (DB URL, model paths, thresholds, CORS, …). |
| `database.py` | SQLAlchemy engine/session factory (`SessionLocal`, `get_db`, `init_db`). Reads `DATABASE_URL` from `settings`. |
| `models.py` | ORM models: `JournalEntry`, `EmotionAnalysis`, `DistortionAnalysis`. |
| `repository.py` | `JournalRepository` — all CRUD and aggregation queries (entries, trends, statistics, counts). |
| `trends.py` | `TrendAnalyzer` — higher-level emotional-evolution, period comparison, weekly summaries, and distortion-frequency analysis built on top of the repository. |
| `utils.py` | Small shared helpers (`utcnow()`). |
| `main.py` | FastAPI app: routes, request/response models, lifespan, analyzer singleton. |
| `streamlit_app.py` | Streamlit UI: new entry, history, trends & insights. |

### `scripts/` — operational helpers
| Script | Responsibility |
| --- | --- |
| `init_db.py` | Create database tables. |
| `check_models.py` | Verify both models load. |

## Import & run conventions

- `app/`, `src/`, and `scripts/` are Python packages (each has `__init__.py`).
- Within a package, use **absolute imports** (`from src.feedback import ...`,
  `from app.repository import ...`).
- **ML command-line tools run as modules from the project root**, e.g.
  `python -m src.train_emotion`, `python -m src.inference --interactive`.
- The API runs as a module (`uvicorn app.main:app`), so it needs no `sys.path`
  manipulation. Two entry points are executed *by file path* and therefore keep
  a small, documented project-root bootstrap: `app/streamlit_app.py` (Streamlit
  puts `app/` on the path, not the root) and the `scripts/*.py` helpers.

## Persistence model

- `JournalEntry` 1─*N* `EmotionAnalysis` and 1─*N* `DistortionAnalysis`
  (cascade delete).
- Only emotions/distortions **above the configured threshold** are stored.
- Timestamps are naive UTC (`app.utils.utcnow`) to match the non-timezone
  `DateTime` columns.

## Request lifecycle examples

- **`POST /api/entries`** → analyze text (ML) → persist entry + detected
  emotions/distortions → return fresh analysis.
- **`GET /api/entries/{id}`** → load entry + related rows → rebuild analysis
  dict (`entry_to_response`) → regenerate insights/recommendations
  (`src/feedback.py`) → return. No model needed.
- **`GET /api/entries/trends` / `/stats`** → aggregation queries in the
  repository. (These static routes are declared *before* `/{entry_id}` so they
  are not captured by the dynamic integer path.)
