"""
Streamlit web interface for MindScribe
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# `streamlit run app/streamlit_app.py` puts the script's own folder (app/) on
# sys.path rather than the project root, so absolute `app.`/`src.` imports would
# fail. Add the project root once so those packages resolve.
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.database import SessionLocal, init_db
from app.repository import JournalRepository
from app.config import settings
from app.trends import TrendAnalyzer
from src.analyzer import JournalAnalyzer

# Page configuration
st.set_page_config(
    page_title="MindScribe - AI Journaling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    try:
        st.session_state.analyzer = JournalAnalyzer(
            emotion_model_path=settings.EMOTION_MODEL_PATH,
            distortion_model_path=settings.DISTORTION_MODEL_PATH,
            emotion_threshold=settings.EMOTION_THRESHOLD,
            distortion_threshold=settings.DISTORTION_THRESHOLD,
            verbose=False
        )
    except Exception as e:
        st.session_state.analyzer = None
        st.error(f"Failed to load models: {e}")

if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .distortion-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff4e6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧠 MindScribe")
    st.markdown("---")
    
    # Model status
    st.subheader("Model Status")
    if st.session_state.analyzer:
        status = st.session_state.analyzer.get_status()
        if status['emotion_classifier']:
            st.success("✓ Emotion Classifier")
        else:
            st.error("✗ Emotion Classifier")
        
        if status['distortion_classifier']:
            st.success("✓ Distortion Classifier")
        else:
            st.error("✗ Distortion Classifier")
    else:
        st.error("Models not loaded")
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    emotion_threshold = st.slider(
        "Emotion Threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.EMOTION_THRESHOLD,
        step=0.05
    )
    
    distortion_threshold = st.slider(
        "Distortion Threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.DISTORTION_THRESHOLD,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("**About MindScribe**")
    st.markdown("AI-powered journaling with emotion tracking and cognitive distortion detection.")

# Main content
st.markdown('<h1 class="main-header">🧠 MindScribe</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Reflective Journaling AI")

# Navigation
page = st.radio(
    "Navigation",
    ["📝 New Entry", "📚 Journal History", "📊 Trends & Insights"],
    horizontal=True
)

if page == "📝 New Entry":
    st.markdown("---")
    st.subheader("Write Your Journal Entry")
    
    # Text input
    journal_text = st.text_area(
        "What's on your mind?",
        height=200,
        placeholder="Write your thoughts, feelings, or experiences here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and journal_text:
        if not st.session_state.analyzer:
            st.error("Models are not loaded. Please check model status in sidebar.")
        else:
            with st.spinner("Analyzing your entry..."):
                # Analyze the text
                result = st.session_state.analyzer.analyze(
                    journal_text,
                    emotion_threshold=emotion_threshold,
                    distortion_threshold=distortion_threshold
                )
                
                # Display results
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Emotions
                if result.get('emotions') and 'top_emotions' in result['emotions']:
                    st.markdown("### 🎭 Detected Emotions")
                    top_emotions = result['emotions']['top_emotions']
                    
                    # Create emotion cards
                    cols = st.columns(min(3, len(top_emotions)))
                    for idx, emotion_data in enumerate(top_emotions[:6]):
                        with cols[idx % 3]:
                            prob = emotion_data['probability']
                            st.metric(
                                label=emotion_data['emotion'].title(),
                                value=f"{prob:.1%}",
                                delta=None
                            )
                            st.progress(prob)
                    
                    # Detected emotions above threshold
                    if result['emotions'].get('detected'):
                        st.markdown("**Emotions above threshold:**")
                        detected = result['emotions']['detected']
                        emotion_list = ", ".join([f"{e['emotion']} ({e['probability']:.1%})" for e in detected])
                        st.info(emotion_list)
                
                # Distortions
                if result.get('distortions') and 'detected' in result['distortions']:
                    st.markdown("### 🧩 Cognitive Distortions")
                    detected = result['distortions']['detected']
                    
                    if detected:
                        for dist in detected:
                            with st.expander(f"⚠️ {dist['name']} ({dist['probability']:.1%})"):
                                st.write(f"**Description:** {dist['description']}")
                                st.progress(dist['probability'])
                    else:
                        st.success("✓ No significant cognitive distortions detected - balanced thinking patterns")
                
                # Insights and Recommendations
                if result.get('insights'):
                    st.markdown("### 💡 Insights")
                    for insight in result['insights']:
                        st.info(f"• {insight}")
                
                if result.get('recommendations'):
                    st.markdown("### 💭 Recommendations")
                    for rec in result['recommendations']:
                        st.success(f"• {rec}")
                
                # Save button
                st.markdown("---")
                if st.button("💾 Save Entry", type="primary"):
                    try:
                        db = SessionLocal()
                        repo = JournalRepository(db)
                        entry = repo.create_entry(
                            text=journal_text,
                            user_id=settings.DEFAULT_USER_ID,
                            analysis_result=result
                        )
                        db.close()
                        st.success(f"✓ Entry saved! (ID: {entry.id})")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving entry: {e}")

elif page == "📚 Journal History":
    st.markdown("---")
    st.subheader("Your Journal Entries")
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Load entries
    try:
        db = SessionLocal()
        repo = JournalRepository(db)
        entries = repo.get_entries(
            user_id=settings.DEFAULT_USER_ID,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time())
        )
        db.close()
        
        if entries:
            st.write(f"Found {len(entries)} entries")
            
            # Display entries
            for entry in entries:
                with st.expander(f"📅 {entry.timestamp.strftime('%Y-%m-%d %H:%M')} - Entry #{entry.id}"):
                    st.write(entry.text)
                    
                    # Quick analysis
                    if st.button(f"🔍 Re-analyze", key=f"analyze_{entry.id}"):
                        if st.session_state.analyzer:
                            with st.spinner("Analyzing..."):
                                result = st.session_state.analyzer.analyze(
                                    entry.text,
                                    emotion_threshold=emotion_threshold,
                                    distortion_threshold=distortion_threshold
                                )
                                
                                if result.get('emotions') and result['emotions'].get('top_emotions'):
                                    st.write("**Top Emotions:**")
                                    for e in result['emotions']['top_emotions'][:3]:
                                        st.write(f"- {e['emotion']}: {e['probability']:.1%}")
                                
                                if result.get('distortions') and result['distortions'].get('detected'):
                                    st.write("**Detected Distortions:**")
                                    for d in result['distortions']['detected']:
                                        st.write(f"- {d['name']}: {d['probability']:.1%}")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"delete_{entry.id}"):
                        try:
                            db = SessionLocal()
                            repo = JournalRepository(db)
                            repo.delete_entry(entry.id, settings.DEFAULT_USER_ID)
                            db.close()
                            st.success("Entry deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting entry: {e}")
        else:
            st.info("No entries found. Start by creating a new entry!")
    
    except Exception as e:
        st.error(f"Error loading entries: {e}")

elif page == "📊 Trends & Insights":
    st.markdown("---")
    st.subheader("Emotional Trends & Insights")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
    )
    
    days_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All time": None
    }
    days = days_map[time_range]
    start_date = datetime.now() - timedelta(days=days) if days else None
    
    try:
        db = SessionLocal()
        repo = JournalRepository(db)

        # AI-generated insights and patterns (TrendAnalyzer)
        # "All time" has no day window, so fall back to a large look-back.
        analysis_days = days if days else 3650
        trend_analyzer = TrendAnalyzer(repo)
        evolution = trend_analyzer.get_emotional_evolution(
            user_id=settings.DEFAULT_USER_ID, days=analysis_days
        )
        emotion_patterns = trend_analyzer.detect_emotional_patterns(
            user_id=settings.DEFAULT_USER_ID, days=analysis_days
        )
        distortion_freq = trend_analyzer.get_distortion_frequency(
            user_id=settings.DEFAULT_USER_ID, days=analysis_days
        )

        st.markdown("### 🧭 AI Insights")
        insight_msgs = [
            msg for msg in (evolution.get("insights", []) + distortion_freq.get("insights", []))
            if "No data available" not in msg
        ]
        if insight_msgs:
            for msg in insight_msgs:
                st.info(f"• {msg}")
        else:
            st.caption("Not enough data yet for automated insights.")

        # Merge + de-duplicate detected patterns (preserve order, drop empty states)
        seen = set()
        pattern_msgs = []
        for pattern in (evolution.get("patterns", []) + emotion_patterns):
            if "No data available" in pattern or pattern in seen:
                continue
            seen.add(pattern)
            pattern_msgs.append(pattern)
        if pattern_msgs:
            st.markdown("**Detected patterns:**")
            for pattern in pattern_msgs:
                st.success(f"• {pattern}")

        st.markdown("---")

        # Emotion statistics
        st.markdown("### 📈 Emotion Statistics")
        stats = repo.get_emotion_statistics(
            user_id=settings.DEFAULT_USER_ID,
            start_date=start_date
        )
        
        if stats['emotions']:
            # Create DataFrame for visualization
            df_emotions = pd.DataFrame(stats['emotions'])
            df_emotions = df_emotions.sort_values('count', ascending=False)
            
            # Bar chart
            fig = px.bar(
                df_emotions.head(10),
                x='emotion',
                y='count',
                title="Most Frequent Emotions",
                labels={'emotion': 'Emotion', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Average probability chart
            fig2 = px.bar(
                df_emotions.head(10),
                x='emotion',
                y='avg_probability',
                title="Average Emotion Intensity",
                labels={'emotion': 'Emotion', 'avg_probability': 'Average Probability'}
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write(f"**Total Entries:** {stats['total_entries']}")
        else:
            st.info("No emotion data available. Create some entries first!")
        
        # Distortion statistics
        st.markdown("### 🧩 Cognitive Distortion Statistics")
        dist_stats = repo.get_distortion_statistics(
            user_id=settings.DEFAULT_USER_ID,
            start_date=start_date
        )
        
        if dist_stats['distortions']:
            df_distortions = pd.DataFrame(dist_stats['distortions'])
            df_distortions = df_distortions.sort_values('count', ascending=False)
            
            # Bar chart
            fig3 = px.bar(
                df_distortions,
                x='name',
                y='count',
                title="Most Common Cognitive Distortions",
                labels={'name': 'Distortion Type', 'count': 'Frequency'}
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            st.write(f"**Entries with Distortions:** {dist_stats['total_entries_with_distortions']}")
        else:
            st.info("No distortion data available.")
        
        # Trends over time
        st.markdown("### 📊 Trends Over Time")
        trends = repo.get_emotion_trends(
            user_id=settings.DEFAULT_USER_ID,
            start_date=start_date
        )
        
        if trends:
            df_trends = pd.DataFrame(trends)
            df_trends['date'] = pd.to_datetime(df_trends['date'])
            
            # Get top emotions for trend
            top_emotions = df_trends.groupby('emotion')['count'].sum().nlargest(5).index.tolist()
            df_trends_filtered = df_trends[df_trends['emotion'].isin(top_emotions)]
            
            fig4 = px.line(
                df_trends_filtered,
                x='date',
                y='avg_probability',
                color='emotion',
                title="Emotion Trends Over Time",
                labels={'date': 'Date', 'avg_probability': 'Average Probability', 'emotion': 'Emotion'}
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No trend data available.")
        
        db.close()
    
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "MindScribe - AI-Powered Reflective Journaling | "
    "Not a replacement for professional mental health care"
    "</div>",
    unsafe_allow_html=True
)
