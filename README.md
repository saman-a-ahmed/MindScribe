# 🧠MindScribe: An Intelligent Reflective Journaling AI

**AI journal that understands your daily text entries, tracks emotional evolution, identifies cognitive biases, and generates psychologically grounded reflections.**

## 🎯 Project Vision

MindScribe combines modern NLP techniques to create a personal AI companion that helps you understand your emotional patterns, recognize cognitive distortions, and gain deeper insights into your mental well-being through intelligent journaling.

## Tech Stack

**Machine Learning & NLP**

-   PyTorch
-   Transformers (Hugging Face)
-   RoBERTa
-   scikit-learn

**Data Processing**

-   pandas
-   NumPy
-   NLTK/spaCy

**Development**

-   Jupyter Notebook
-   Python 3.8+

**Deployment**

-   Streamlit/Gradio
-   FastAPI
-   React/Next.js
-   PostgreSQL

## Run Locally

### Prerequisites

-   Python 3.8 or higher
-   pip
-   Git

### Datasets

**Emotion Classification:**
This project uses the **GoEmotions** dataset from Google Research:
-   58,000+ Reddit comments
-   27 emotion categories
-   Multi-label annotations

The dataset will be automatically downloaded during preprocessing, or you can manually download it from [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).

**Cognitive Distortion Detection:**
-   Synthetic dataset generated using pattern-based generation
-   12 cognitive distortion types (based on CBT principles)
-   Multi-label annotations
-   Generated examples for training and evaluation

# Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mindscribe.git
cd mindscribe

# Create and activate virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

#--------------Emotion Classification--------------

# Preprocess the emotion dataset
python src/preprocessing.py

# Train the emotion classifier (default settings)
python src/train_emotion.py

# Train with custom parameters
python src/train_emotion.py --epochs 5 --batch_size 16 --learning_rate 2e-5

# Run inference on emotions
python src/inference.py              # Sanity checks
python src/inference.py --interactive  # Interactive mode

#--------------Cognitive Distortion Detection--------------

# Step 1: Generate synthetic dataset
python src/generate_distortion_data.py

# Step 2: Preprocess the dataset
python src/preprocess_distortions.py

# Step 3: Train the distortion classifier (default settings)
python src/train_distortion.py

# Train with custom parameters
python src/train_distortion.py --epochs 10 --batch_size 16 --learning_rate 1e-5

# Step 4: Run inference on cognitive distortions
python src/inference_distortion.py              # Sanity checks
python src/inference_distortion.py --interactive  # Interactive mode
```

## Features

### ✅ Emotion Classification
- Detects 28 emotion categories from text
- Multi-label classification (can detect multiple emotions)
- Trained on GoEmotions dataset
- Real-time inference support

### ✅ Cognitive Distortion Detection
- Detects 12 cognitive distortion types:
  - All-or-Nothing Thinking
  - Overgeneralization
  - Mental Filter
  - Discounting the Positive
  - Mind Reading
  - Fortune Telling
  - Catastrophizing
  - Minimization
  - Emotional Reasoning
  - Should Statements
  - Labeling
  - Personalization
- Multi-label classification (can detect multiple distortions)
- Based on CBT (Cognitive Behavioral Therapy) principles
- Synthetic training data with pattern-based generation

## Cognitive Distortions

Cognitive distortions are systematic patterns of thinking that deviate from logic or reality. This component helps identify these patterns in journal entries:

- **All-or-Nothing Thinking**: Seeing things in extreme categories with no middle ground
- **Overgeneralization**: Making broad conclusions from single events
- **Mental Filter**: Focusing exclusively on negatives, ignoring positives
- **Discounting the Positive**: Rejecting positive experiences as "not counting"
- **Mind Reading**: Assuming you know what others think without evidence
- **Fortune Telling**: Predicting negative outcomes with certainty
- **Catastrophizing**: Expecting or imagining the worst-case scenario
- **Minimization**: Inappropriately shrinking the importance of positive events
- **Emotional Reasoning**: Believing feelings equal facts
- **Should Statements**: Rigid rules about how you/others "should" behave
- **Labeling**: Assigning global negative labels to self/others
- **Personalization**: Blaming yourself for things outside your control

## 🚀 Deployment

### Quick Start with Docker

The easiest way to run MindScribe is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/mindscribe.git
cd mindscribe

# Start all services (database, API, Streamlit)
docker-compose up -d

# Initialize database
docker-compose exec api python scripts/init_db.py

# Check model status
docker-compose exec api python scripts/check_models.py
```

Access the application:
- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Local Development Setup

1. **Install Dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Train Models** (if not already trained)
```bash
# Emotion classifier
python src/preprocessing.py
python src/train_emotion.py

# Distortion classifier
python src/generate_distortion_data.py
python src/preprocess_distortions.py
python src/train_distortion.py
```

3. **Initialize Database**
```bash
python scripts/init_db.py
python scripts/check_models.py
```

4. **Run the Application**

**Option A: Streamlit (Recommended for UI)**
```bash
streamlit run app/streamlit_app.py
```

**Option B: FastAPI Backend**
```bash
python -m uvicorn app.main:app --reload
```

### Environment Configuration

Create a `.env` file (see `.env.example` for template):

```env
DATABASE_URL=sqlite:///./mindscribe.db  # or postgresql://user:pass@localhost/mindscribe
EMOTION_MODEL_PATH=models/emotion_classifier
DISTORTION_MODEL_PATH=models/distortion_classifier
EMOTION_THRESHOLD=0.3
DISTORTION_THRESHOLD=0.3
API_HOST=0.0.0.0
API_PORT=8000
```

## 📡 API Documentation

### Endpoints

#### Health Check
- `GET /` - API information
- `GET /health` - Health check

#### Journal Entries
- `POST /api/entries` - Create new entry with analysis
- `GET /api/entries` - List entries (with pagination, date filters)
- `GET /api/entries/{id}` - Get specific entry
- `DELETE /api/entries/{id}` - Delete entry

#### Analysis
- `POST /api/analyze` - Analyze text without saving

#### Statistics & Trends
- `GET /api/entries/trends` - Get emotional trends over time
- `GET /api/entries/stats` - Get aggregated emotion statistics
- `GET /api/distortions/stats` - Get distortion statistics

### Example API Usage

```python
import requests

# Create a journal entry
response = requests.post("http://localhost:8000/api/entries", json={
    "text": "I feel anxious about tomorrow's presentation. I'm definitely going to fail.",
    "user_id": "default"
})
entry = response.json()

# Get all entries
entries = requests.get("http://localhost:8000/api/entries").json()

# Analyze text without saving
analysis = requests.post("http://localhost:8000/api/analyze", 
                         params={"text": "Today was a good day!"}).json()

# Get trends
trends = requests.get("http://localhost:8000/api/entries/trends",
                     params={"days": 30}).json()
```

## 🎨 Web Interface

The Streamlit interface provides:

1. **New Entry**: Write and analyze journal entries in real-time
2. **Journal History**: Browse past entries with filters
3. **Trends & Insights**: Visualize emotional patterns and distortion frequencies

Features:
- Real-time emotion and distortion detection
- Interactive charts and visualizations
- Entry management (view, delete)
- Trend analysis over time

## 📁 Project Structure

```
MindScribe/
├── app/                    # Application code
│   ├── main.py             # FastAPI application
│   ├── streamlit_app.py     # Streamlit UI
│   ├── config.py           # Configuration
│   ├── database.py         # Database connection
│   ├── models.py           # SQLAlchemy models
│   ├── repository.py       # Data access layer
│   └── trends.py           # Trend analysis
├── src/                     # ML models and inference
│   ├── analyzer.py         # Unified analysis service
│   ├── inference.py        # Emotion classifier
│   ├── inference_distortion.py  # Distortion classifier
│   └── ...                 # Training scripts
├── scripts/                 # Utility scripts
│   ├── init_db.py          # Database initialization
│   └── check_models.py     # Model verification
├── models/                  # Trained models
├── data/                    # Datasets
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── requirements.txt        # Python dependencies
```

## 🔧 Development

### Running Tests

```bash
# Check models
python scripts/check_models.py

# Test API
curl http://localhost:8000/health
```

### Database Migrations

The database schema is automatically created on first run. To reset:

```bash
python scripts/init_db.py
```

## 📊 Features

### ✅ Emotion Classification
- Detects 28 emotion categories from text
- Multi-label classification (can detect multiple emotions)
- Trained on GoEmotions dataset
- Real-time inference support

### ✅ Cognitive Distortion Detection
- Detects 12 cognitive distortion types:
  - All-or-Nothing Thinking
  - Overgeneralization
  - Mental Filter
  - Discounting the Positive
  - Mind Reading
  - Fortune Telling
  - Catastrophizing
  - Minimization
  - Emotional Reasoning
  - Should Statements
  - Labeling
  - Personalization
- Multi-label classification (can detect multiple distortions)
- Based on CBT (Cognitive Behavioral Therapy) principles
- Synthetic training data with pattern-based generation

### ✅ Web Application
- Streamlit-based user interface
- FastAPI REST backend
- PostgreSQL/SQLite database support
- Trend visualization and analytics
- Entry management and history

## 🐳 Docker Deployment

### Production Deployment

1. **Build and run with Docker Compose:**
```bash
docker-compose up -d
```

2. **Initialize database:**
```bash
docker-compose exec api python scripts/init_db.py
```

3. **Check logs:**
```bash
docker-compose logs -f
```

### Custom Configuration

Edit `docker-compose.yml` to customize:
- Database credentials
- Port mappings
- Volume mounts
- Environment variables

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]

## ⚠️ Disclaimer

MindScribe is a reflective tool designed to help users understand their emotional patterns and thought processes. It is **not a replacement for professional mental health care**. Always seek professional help when needed.

**Note**: This project is under active development. Both emotion classification and cognitive distortion detection components are complete and functional. Additional features are being built incrementally.