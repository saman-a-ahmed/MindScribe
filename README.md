# 🧠 MindScribe: An Intelligent Reflective Journaling AI

**An AI journal that understands your daily text entries, tracks emotional evolution, identifies cognitive distortions, and surfaces psychologically grounded reflections.**

## 🎯 Project Vision

MindScribe combines modern NLP techniques into a personal AI companion that helps you understand your emotional patterns, recognize cognitive distortions, and gain deeper insight into your mental well-being through intelligent journaling.

## ✨ Features

### Emotion Classification
- Detects **28 emotion categories** from text (GoEmotions taxonomy)
- Multi-label classification — multiple emotions per entry
- RoBERTa-based, trained on the GoEmotions dataset
- Real-time inference

### Cognitive Distortion Detection
- Detects **12 cognitive distortion types** grounded in CBT (see [taxonomy](#-cognitive-distortions))
- Multi-label classification
- Trained on a synthetic, pattern-based dataset

### Web Application
- **Streamlit** UI: write & analyze entries, browse history, visualize trends and AI insights
- **FastAPI** REST backend with automatic analysis on entry creation
- **SQLite / PostgreSQL** persistence
- Trend visualization and pattern detection (`TrendAnalyzer`)

## 🧱 Tech Stack

| Area | Tools |
| --- | --- |
| ML / NLP | PyTorch, Transformers (Hugging Face), RoBERTa, scikit-learn |
| Data | pandas, NumPy, Hugging Face `datasets` |
| Backend | FastAPI, Uvicorn, SQLAlchemy |
| Frontend | Streamlit, Plotly |
| Database | PostgreSQL / SQLite |
| Config | pydantic, python-dotenv |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and data flow.

## 📁 Project Structure

```
MindScribe/
├── app/                     # Web application
│   ├── main.py              # FastAPI application
│   ├── streamlit_app.py     # Streamlit UI
│   ├── config.py            # Settings (single source of truth)
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # ORM models
│   ├── repository.py        # Data access layer
│   ├── trends.py            # Trend analysis (TrendAnalyzer)
│   └── utils.py             # Shared helpers (utcnow)
├── src/                     # Machine learning
│   ├── analyzer.py          # Unified JournalAnalyzer
│   ├── feedback.py          # Model-free insights/recommendations
│   ├── cognitive_distortions.py  # Distortion taxonomy
│   ├── inference.py         # Emotion classifier
│   ├── inference_distortion.py   # Distortion classifier
│   ├── preprocessing.py / train_emotion.py
│   └── generate_distortion_data.py / preprocess_distortions.py / train_distortion.py
├── scripts/                 # Operational helpers (init_db, check_models)
├── tests/                   # Pytest suite (non-ML units)
├── models/                  # Trained models
├── data/                    # Datasets
├── docs/ARCHITECTURE.md     # Architecture overview
├── Dockerfile / docker-compose.yml
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## 🚀 Quick Start (Docker)

The easiest way to run the full stack (database, API, Streamlit):

```bash
git clone https://github.com/yourusername/mindscribe.git
cd mindscribe

docker-compose up -d                                   # start all services
docker-compose exec api python scripts/init_db.py      # initialize the database
docker-compose exec api python scripts/check_models.py # verify models load
```

- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- API docs: http://localhost:8000/docs

## 🛠️ Local Development

### Prerequisites
- Python 3.10+ (3.8+ for the ML scripts; the app targets 3.10)
- pip and Git

### 1. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Optional: copy and edit environment configuration
cp .env.example .env
```

### 2. Train the models (if not already present)

> All ML scripts are run **as modules from the project root** (`python -m src.<name>`).

```bash
# --- Emotion classifier ---
python -m src.preprocessing            # download + preprocess GoEmotions
python -m src.train_emotion            # train (defaults)
python -m src.train_emotion --epochs 5 --batch_size 16 --learning_rate 2e-5
python -m src.inference                # sanity checks
python -m src.inference --interactive  # interactive mode

# --- Cognitive distortion classifier ---
python -m src.generate_distortion_data # generate synthetic dataset
python -m src.preprocess_distortions   # preprocess
python -m src.train_distortion         # train (defaults)
python -m src.train_distortion --epochs 10 --batch_size 16 --learning_rate 1e-5
python -m src.inference_distortion                # sanity checks
python -m src.inference_distortion --interactive  # interactive mode
```

### 3. Initialize the database

```bash
python scripts/init_db.py
python scripts/check_models.py
```

### 4. Run the application

```bash
# Option A — Streamlit UI (recommended)
streamlit run app/streamlit_app.py

# Option B — FastAPI backend
python -m uvicorn app.main:app --reload
```

## ⚙️ Configuration

Configuration is read from environment variables / a `.env` file (see
[`.env.example`](.env.example) for all options and defaults):

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./mindscribe.db` | SQLite or PostgreSQL connection string |
| `EMOTION_MODEL_PATH` | `models/emotion_classifier` | Path to the emotion model |
| `DISTORTION_MODEL_PATH` | `models/distortion_classifier` | Path to the distortion model |
| `EMOTION_THRESHOLD` / `DISTORTION_THRESHOLD` | `0.3` | Detection probability cut-offs |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API bind address |
| `CORS_ORIGINS` | `http://localhost:8501,http://localhost:3000` | Allowed origins (comma-separated) |

## 📡 API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | API info + model status |
| `GET` | `/health` | Health check |
| `POST` | `/api/entries` | Create an entry (runs analysis, persists results) |
| `GET` | `/api/entries` | List entries (pagination + date filters) |
| `GET` | `/api/entries/trends` | Emotional trends over time |
| `GET` | `/api/entries/stats` | Aggregated emotion statistics |
| `GET` | `/api/entries/{id}` | Get a single entry |
| `DELETE` | `/api/entries/{id}` | Delete an entry |
| `GET` | `/api/distortions/stats` | Aggregated distortion statistics |
| `POST` | `/api/analyze` | Analyze text without saving |

### Example

```python
import requests

# Create a journal entry (analysis is run and stored automatically)
entry = requests.post("http://localhost:8000/api/entries", json={
    "text": "I feel anxious about tomorrow's presentation. I'm definitely going to fail.",
    "user_id": "default",
}).json()

# List entries (analysis is rebuilt from stored data — no re-inference)
entries = requests.get("http://localhost:8000/api/entries").json()

# Analyze text without saving
analysis = requests.post("http://localhost:8000/api/analyze",
                         json={"text": "Today was a good day!"}).json()

# Trends
trends = requests.get("http://localhost:8000/api/entries/trends",
                      params={"emotion": "joy"}).json()
```

## 🎨 Web Interface

The Streamlit app provides three views:

1. **New Entry** — write and analyze entries in real time
2. **Journal History** — browse and manage past entries with date filters
3. **Trends & Insights** — emotion/distortion charts plus `TrendAnalyzer` insights and detected patterns

## 🧠 Cognitive Distortions

Cognitive distortions are systematic patterns of thinking that deviate from logic or reality. MindScribe detects these 12 CBT-based types:

- **All-or-Nothing Thinking** — seeing things in extreme categories with no middle ground
- **Overgeneralization** — broad conclusions from single events
- **Mental Filter** — focusing exclusively on negatives
- **Discounting the Positive** — rejecting positives as "not counting"
- **Mind Reading** — assuming you know what others think
- **Fortune Telling** — predicting negative outcomes with certainty
- **Catastrophizing** — expecting the worst-case scenario
- **Minimization** — shrinking the importance of positive events
- **Emotional Reasoning** — believing feelings equal facts
- **Should Statements** — rigid rules about how you/others "should" behave
- **Labeling** — assigning global negative labels
- **Personalization** — blaming yourself for things outside your control

## 🧪 Testing

A lightweight pytest suite covers the non-ML pieces (repository, trends, feedback, taxonomy, config) and runs without PyTorch/Transformers:

```bash
pip install -r requirements-dev.txt
pytest
```

## 🐳 Docker Deployment

```bash
docker-compose up -d                                # build & start
docker-compose exec api python scripts/init_db.py   # initialize DB
docker-compose logs -f                              # follow logs
```

Edit `docker-compose.yml` to customize database credentials, ports, volumes, and environment variables.

## 🤝 Contributing

Contributions are welcome — please open an issue or submit a pull request.

## 📝 License

[Add your license here]

## ⚠️ Disclaimer

MindScribe is a reflective tool to help you understand your emotional patterns and thought processes. It is **not a replacement for professional mental health care**. Always seek professional help when needed.
