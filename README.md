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

### Dataset

This project uses the **GoEmotions** dataset from Google Research:

-   58,000+ Reddit comments
-   27 emotion categories
-   Multi-label annotations

The dataset will be automatically downloaded during preprocessing, or you can manually download it from [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).

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

# Preprocess the dataset
python src/preprocessing.py

#--------------Train the emotion classifier--------------

# Train with default settings
python src/train_emotion.py

# Train with custom parameters
python src/train_emotion.py --epochs 5 --batch_size 16 --learning_rate 2e-5
```


**Note**: This project is under active development. The emotion classification component is complete and functional. Additional features are being built incrementally.

**Disclaimer**: MindScribe is a reflective tool, not a replacement for professional mental health care. Always seek professional help when needed.