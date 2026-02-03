"""
Digital Witness - Phase 2 Demo UI

Streamlit web interface with dual-mode video analysis:
- Real-Time Mode: RTSP camera feed with interactive POS simulator
- Upload Mode: Pre-recorded video with manual POS entry

Run with: streamlit run app.py
"""
import streamlit as st
import tempfile
import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import POS_PRODUCT_CATALOG, ALERTS_STORAGE_DIR
from src.pos import POSSimulator
from src.alerts import AlertManager, AlertStatus, NotificationService

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Digital Witness - Main Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme-aware CSS
def get_theme_css():
    """Generate CSS based on current theme."""
    is_dark = st.session_state.theme == 'dark'

    if is_dark:
        bg_primary = '#0e1117'
        bg_secondary = '#1a1d24'
        bg_card = '#262730'
        text_primary = '#fafafa'
        text_secondary = '#b0b0b0'
        border_color = '#3d4149'
        sidebar_bg = '#1a1d24'
    else:
        bg_primary = '#ffffff'
        bg_secondary = '#f8f9fa'
        bg_card = '#ffffff'
        text_primary = '#1e3a5f'
        text_secondary = '#666666'
        border_color = '#dee2e6'
        sidebar_bg = '#f8f9fa'

    return f"""
    <style>
        /* Theme variables */
        :root {{
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-card: {bg_card};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
        }}

        /* Page navigation */
        .page-nav {{
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 15px;
            background: {bg_secondary};
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .page-nav-btn {{
            padding: 10px 25px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .page-nav-btn-active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }}
        .page-nav-btn-inactive {{
            background: {bg_card};
            color: {text_secondary};
            border: 1px solid {border_color};
        }}
        .page-nav-btn-inactive:hover {{
            background: {border_color};
        }}

        /* Main header styles */
        .main-title {{
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(120deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 0;
        }}
        .subtitle {{
            font-size: 1.3rem;
            color: {text_secondary};
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }}

        /* Metric cards */
        .metric-container {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 1rem;
            opacity: 0.9;
        }}

        /* Alert boxes */
        .alert-critical {{
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        .alert-high {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        .alert-medium {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: #333;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        .alert-low {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #333;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        .alert-none {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}

        /* Info boxes */
        .info-box {{
            background-color: {bg_secondary};
            border-left: 4px solid #667eea;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: {text_primary};
        }}

        /* Section headers */
        .section-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {text_primary};
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem 0;
        }}

        /* Progress bar customization */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }}

        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}

        /* Card container */
        .card {{
            background: {bg_card};
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin: 1rem 0;
            border: 1px solid {border_color};
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {bg_secondary};
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        /* Mode selector styling */
        .mode-active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }}

        /* POS Simulator styling */
        .pos-item-btn {{
            background: {bg_secondary};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            transition: all 0.2s;
        }}
        .pos-item-btn:hover {{
            background: {border_color};
            transform: translateY(-2px);
        }}

        /* Real-time stats */
        .realtime-stat {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}

        /* Connection status */
        .connection-active {{
            border-left: 4px solid #28a745;
            padding-left: 10px;
        }}
        .connection-inactive {{
            border-left: 4px solid #ffc107;
            padding-left: 10px;
        }}

        /* Theme toggle */
        .theme-toggle {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: {bg_secondary};
            border-radius: 8px;
            margin-bottom: 15px;
        }}
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)


# ============== HELPER FUNCTIONS ==============

def check_model_exists():
    """Check if the trained model exists."""
    from src.config import BEHAVIOR_MODEL_PATH
    return BEHAVIOR_MODEL_PATH.exists()


def load_model_metrics():
    """Load model performance metrics from the info file."""
    from src.config import MODELS_DIR
    info_path = MODELS_DIR / "behavior_classifier_info.json"

    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return None


def get_product_catalog():
    """Get the product catalog for POS data - uses config for consistency."""
    return POS_PRODUCT_CATALOG


def init_realtime_session_state():
    """Initialize session state for real-time mode."""
    if 'pos_simulator' not in st.session_state:
        st.session_state.pos_simulator = POSSimulator()

    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager(ALERTS_STORAGE_DIR)

    if 'notification_service' not in st.session_state:
        st.session_state.notification_service = NotificationService()

    if 'realtime_running' not in st.session_state:
        st.session_state.realtime_running = False

    if 'video_mode' not in st.session_state:
        st.session_state.video_mode = "upload"  # "upload" or "realtime"

    if 'rtsp_url' not in st.session_state:
        st.session_state.rtsp_url = ""

    if 'pos_cart' not in st.session_state:
        st.session_state.pos_cart = []

    if 'pos_session_active' not in st.session_state:
        st.session_state.pos_session_active = False


def create_pos_data(items_list, payment_method="card"):
    """Create POS transaction data from selected items."""
    if not items_list:
        return {"transactions": []}

    total = sum(item['price'] * item.get('quantity', 1) for item in items_list)

    transaction = {
        "transaction_id": f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "items": [
            {
                "sku": item['sku'],
                "name": item['name'],
                "quantity": item.get('quantity', 1),
                "price": item['price']
            }
            for item in items_list
        ],
        "total": round(total, 2),
        "payment_method": payment_method
    }

    return {"transactions": [transaction]}


def analyze_video(video_path, pos_data=None, progress_callback=None):
    """
    Analyze a video file and return results.
    """
    from src.video import VideoLoader
    from src.pose import PoseEstimator, FeatureExtractor, BehaviorClassifier
    from src.config import SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE

    results = {
        'success': False,
        'error': None,
        'video_info': {},
        'behavior_events': [],
        'prediction': None,
        'confidence': 0.0,
        'risk_level': 'LOW',
        'processing_stats': {},
        'pos_data': pos_data,
        'intent_analysis': None
    }

    try:
        # Step 1: Load video
        if progress_callback:
            progress_callback(0.1, "Loading video...")

        with VideoLoader(video_path) as loader:
            metadata = loader.metadata
            results['video_info'] = {
                'filename': Path(video_path).name,
                'duration': f"{metadata.duration:.1f}s",
                'duration_seconds': metadata.duration,
                'fps': metadata.fps,
                'resolution': f"{metadata.width}x{metadata.height}",
                'total_frames': metadata.frame_count
            }

            # Step 2: Pose estimation
            if progress_callback:
                progress_callback(0.2, "Initializing pose estimation...")

            pose_estimator = PoseEstimator()
            feature_extractor = FeatureExtractor(
                window_size=SLIDING_WINDOW_SIZE,
                window_stride=SLIDING_WINDOW_STRIDE
            )

            pose_results = []
            frame_count = 0
            total_frames = metadata.frame_count

            if progress_callback:
                progress_callback(0.25, "Processing frames...")

            # Process every 2nd frame for speed
            for frame_num, frame in loader.frames(step=2):
                pose_result = pose_estimator.process_frame(frame, frame_num, metadata.fps)
                pose_results.append(pose_result)
                frame_count += 1

                if progress_callback and frame_count % 20 == 0:
                    progress = 0.25 + (0.5 * (frame_num / total_frames))
                    progress_callback(progress, f"Processing frame {frame_num}/{total_frames}...")

            pose_estimator.close()

            # Step 3: Feature extraction
            if progress_callback:
                progress_callback(0.75, "Extracting features...")

            pose_features = feature_extractor.extract_from_sequence(pose_results)

            if not pose_features:
                results['error'] = "Could not extract features from video"
                return results

            # Step 4: Classification
            if progress_callback:
                progress_callback(0.85, "Classifying behavior...")

            classifier = BehaviorClassifier()

            # Get predictions for all windows
            feature_matrix = np.vstack([pf.features for pf in pose_features])
            predictions = classifier.predict(feature_matrix)
            probabilities = classifier.predict_proba(feature_matrix)

            # Aggregate results
            shoplifting_count = np.sum(predictions == 1)
            normal_count = np.sum(predictions == 0)
            total_windows = len(predictions)

            # Calculate overall confidence
            shoplifting_ratio = shoplifting_count / total_windows if total_windows > 0 else 0
            avg_shoplifting_prob = np.mean(probabilities[:, 1]) if probabilities.shape[1] > 1 else 0

            # Determine final prediction
            if shoplifting_ratio > 0.5:
                results['prediction'] = 'SHOPLIFTING DETECTED'
                results['confidence'] = avg_shoplifting_prob
            else:
                results['prediction'] = 'NORMAL BEHAVIOR'
                results['confidence'] = 1 - avg_shoplifting_prob

            # Determine risk level
            if avg_shoplifting_prob >= 0.7:
                results['risk_level'] = 'HIGH'
            elif avg_shoplifting_prob >= 0.5:
                results['risk_level'] = 'MEDIUM'
            elif avg_shoplifting_prob >= 0.3:
                results['risk_level'] = 'LOW'
            else:
                results['risk_level'] = 'NONE'

            # Create behavior timeline
            behavior_events = []
            for i, (pf, pred, prob) in enumerate(zip(pose_features, predictions, probabilities)):
                event = {
                    'window': i + 1,
                    'start_time': pf.window_start_time,
                    'end_time': pf.window_end_time,
                    'start_time_str': f"{pf.window_start_time:.1f}s",
                    'end_time_str': f"{pf.window_end_time:.1f}s",
                    'behavior': classifier.classes[pred],
                    'confidence': float(np.max(prob)),
                    'shoplifting_prob': float(prob[1]) if len(prob) > 1 else 0
                }
                behavior_events.append(event)

            results['behavior_events'] = behavior_events
            results['processing_stats'] = {
                'frames_processed': frame_count,
                'poses_detected': sum(1 for p in pose_results if p.landmarks is not None),
                'feature_windows': len(pose_features),
                'shoplifting_windows': int(shoplifting_count),
                'normal_windows': int(normal_count),
                'shoplifting_ratio': shoplifting_ratio,
                'avg_shoplifting_probability': avg_shoplifting_prob
            }

            # Step 5: Intent scoring if POS data provided
            if progress_callback:
                progress_callback(0.95, "Calculating intent score...")

            if pos_data and pos_data.get('transactions'):
                results['intent_analysis'] = calculate_intent_score(
                    behavior_events,
                    pos_data,
                    metadata.duration
                )

            if progress_callback:
                progress_callback(1.0, "Analysis complete!")

            results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results


def calculate_intent_score(behavior_events, pos_data, video_duration):
    """Calculate intent score based on behavior and POS data."""
    # Count suspicious behaviors
    shoplifting_events = [e for e in behavior_events if e['behavior'] == 'shoplifting']

    # Check POS billing
    transactions = pos_data.get('transactions', [])
    total_billed_items = sum(
        len(t.get('items', []))
        for t in transactions
    )

    # Simple scoring logic
    shoplifting_ratio = len(shoplifting_events) / len(behavior_events) if behavior_events else 0

    # Calculate component scores
    behavior_score = min(1.0, shoplifting_ratio * 1.5)
    billing_score = 0.0 if total_billed_items > 0 else 0.5

    # Combined score
    intent_score = (behavior_score * 0.7 + billing_score * 0.3)

    return {
        'score': round(intent_score, 2),
        'behavior_score': round(behavior_score, 2),
        'billing_score': round(billing_score, 2),
        'suspicious_windows': len(shoplifting_events),
        'total_windows': len(behavior_events),
        'items_billed': total_billed_items,
        'explanation': generate_intent_explanation(intent_score, shoplifting_events, total_billed_items)
    }


def generate_intent_explanation(score, shoplifting_events, items_billed):
    """Generate human-readable explanation."""
    lines = []

    if score >= 0.7:
        lines.append("HIGH RISK: Multiple indicators suggest potential shoplifting.")
    elif score >= 0.5:
        lines.append("MEDIUM RISK: Some concerning patterns detected.")
    elif score >= 0.3:
        lines.append("LOW RISK: Minor anomalies detected.")
    else:
        lines.append("MINIMAL RISK: Behavior appears normal.")

    if shoplifting_events:
        lines.append(f"Detected {len(shoplifting_events)} suspicious behavior window(s).")

    if items_billed == 0:
        lines.append("No items were billed in the POS transaction.")
    else:
        lines.append(f"{items_billed} item(s) billed at checkout.")

    lines.append("")
    lines.append("NOTE: This is an advisory assessment. Final determination requires human review.")

    return "\n".join(lines)


# ============== UI COMPONENTS ==============

def render_theme_toggle():
    """Render theme toggle at the top of the page."""
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        # Theme toggle using checkbox styled as switch
        is_dark = st.toggle(
            "Dark Mode",
            value=st.session_state.theme == 'dark',
            key="theme_toggle_main"
        )
        new_theme = 'dark' if is_dark else 'light'
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()


def render_header():
    """Render the main header."""
    render_theme_toggle()
    st.markdown('<h1 class="main-title">Digital Witness</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Bias-Aware, Explainable Retail Security Assistant</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)

        # Navigation
        st.markdown("### Navigation")
        st.page_link("app.py", label="Main Dashboard", icon="📊")
        st.page_link("pages/Security_Manager.py", label="Security Manager", icon="🛡️")

        st.markdown("---")

        st.markdown("### About Digital Witness")
        st.markdown("""
        An AI-powered security assistant that detects potential shoplifting through:

        - **Pose Analysis**: Body movement patterns
        - **Behavior Classification**: ML-based detection
        - **POS Correlation**: Transaction verification
        - **Explainable Results**: Clear reasoning
        """)

        st.markdown("---")

        # Model Status
        st.markdown("### System Status")
        if check_model_exists():
            st.success("Model: Ready")
            metrics = load_model_metrics()
            if metrics:
                acc = metrics['metrics']['accuracy']
                st.metric("Model Accuracy", f"{acc:.1%}")
        else:
            st.error("Model: Not Found")
            st.warning("Run `python train.py` first")

        st.markdown("---")

        # Email notification status
        st.markdown("### Notifications")
        try:
            if st.session_state.notification_service.is_email_configured:
                st.success("Email: Configured")
            else:
                st.warning("Email: Not configured")
                st.caption("Set EMAIL_SENDER and EMAIL_PASSWORD env vars")
        except Exception:
            st.info("Email: Check config")

        st.markdown("---")

        st.markdown("### Important Notice")
        st.info("""
        This is an **advisory system**.
        All alerts require human validation.
        The system does NOT determine guilt.
        """)

        st.markdown("---")
        st.markdown("##### Final Year Project - 2026")


def render_model_performance_tab():
    """Render the model performance tab."""
    metrics = load_model_metrics()

    if not metrics:
        st.warning("No model metrics found. Please train the model first using `python train.py`")
        return

    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    # Training info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Date", metrics['training_date'][:10])
    with col2:
        st.metric("Training Samples", f"{metrics['n_train_samples']:,}")
    with col3:
        st.metric("Test Samples", f"{metrics['n_test_samples']:,}")

    st.markdown("---")

    # Main metrics
    st.markdown("### Classification Metrics")

    col1, col2, col3, col4 = st.columns(4)

    m = metrics['metrics']
    with col1:
        st.metric("Accuracy", f"{m['accuracy']:.1%}", help="Overall correct predictions")
    with col2:
        st.metric("Precision", f"{m['precision']:.1%}", help="True positives / All predicted positives")
    with col3:
        st.metric("Recall", f"{m['recall']:.1%}", help="True positives / All actual positives")
    with col4:
        st.metric("F1 Score", f"{m['f1_score']:.1%}", help="Harmonic mean of precision and recall")

    st.markdown("---")

    # Cross-validation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cross-Validation")
        cv_mean = m['cv_accuracy_mean']
        cv_std = m['cv_accuracy_std']
        st.metric(
            "CV Accuracy (5-Fold)",
            f"{cv_mean:.1%}",
            delta=f"± {cv_std:.1%}",
            help="Average accuracy across 5 cross-validation folds"
        )

        # CV visualization
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = cv_mean * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CV Accuracy %"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 70], 'color': "#fff3cd"},
                    {'range': [70, 85], 'color': "#d4edda"},
                    {'range': [85, 100], 'color': "#c3e6cb"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Confusion Matrix")

        conf_matrix = np.array(metrics['confusion_matrix'])
        classes = metrics['classes']

        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 20},
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.markdown("### Feature Importance")

    importance = metrics['feature_importance']
    df_importance = pd.DataFrame([
        {'Feature': k, 'Importance': v}
        for k, v in importance.items()
    ])

    import plotly.express as px
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model Configuration
    st.markdown("---")
    st.markdown("### Model Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Model Parameters**")
        st.json(metrics['model_params'])
    with col2:
        st.markdown("**Processing Config**")
        st.json(metrics['processing_config'])
    with col3:
        st.markdown("**Classes**")
        for i, c in enumerate(metrics['classes']):
            st.write(f"{i}: {c}")


def render_analysis_results(results):
    """Render the analysis results."""
    if not results['success']:
        st.error(f"Analysis failed: {results['error']}")
        if 'traceback' in results:
            with st.expander("Error Details"):
                st.code(results['traceback'])
        return

    import plotly.graph_objects as go
    import plotly.express as px

    # Main prediction banner
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    prediction = results['prediction']
    risk = results['risk_level']
    confidence = results['confidence']

    # Result banner
    if 'SHOPLIFTING' in prediction:
        alert_class = 'alert-high' if risk == 'HIGH' else 'alert-medium'
        st.markdown(f"""
        <div class='{alert_class}'>
            <h2 style='margin:0; font-size:1.8rem;'>⚠️ {prediction}</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Confidence: {confidence:.1%} | Risk Level: {risk}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='alert-none'>
            <h2 style='margin:0; font-size:1.8rem;'>✓ {prediction}</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Confidence: {confidence:.1%} | Risk Level: {risk}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    stats = results['processing_stats']
    with col1:
        st.metric("Frames Analyzed", f"{stats['frames_processed']:,}")
    with col2:
        st.metric("Poses Detected", f"{stats['poses_detected']:,}")
    with col3:
        st.metric("Analysis Windows", f"{stats['feature_windows']}")
    with col4:
        st.metric("Shoplifting Ratio", f"{stats['shoplifting_ratio']:.1%}")

    st.markdown("---")

    # Intent Analysis (if available)
    if results.get('intent_analysis'):
        st.markdown("### Intent Score Analysis")

        intent = results['intent_analysis']

        col1, col2 = st.columns([1, 2])

        with col1:
            # Intent gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = intent['score'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Intent Risk Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},
                        {'range': [30, 50], 'color': "#fff3cd"},
                        {'range': [50, 70], 'color': "#ffe4b2"},
                        {'range': [70, 100], 'color': "#ffcccc"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Explanation:**")
            st.info(intent['explanation'])

            st.markdown("**Score Components:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Behavior Score", f"{intent['behavior_score']:.0%}")
            with col_b:
                st.metric("Billing Score", f"{intent['billing_score']:.0%}")
            with col_c:
                st.metric("Items Billed", intent['items_billed'])

        st.markdown("---")

    # Behavior Timeline
    st.markdown("### Behavior Timeline")

    events = results['behavior_events']
    if events:
        # Create timeline chart
        df_events = pd.DataFrame(events)

        fig = go.Figure()

        # Add bars for each window
        colors = []
        for e in events:
            if e['behavior'] == 'shoplifting':
                colors.append('#ff4b4b')
            else:
                colors.append('#00c853')

        fig.add_trace(go.Bar(
            y=[f"W{e['window']}" for e in events],
            x=[e['shoplifting_prob'] for e in events],
            orientation='h',
            marker_color=colors,
            text=[f"{e['shoplifting_prob']:.0%}" for e in events],
            textposition='inside',
            hovertemplate="<b>Window %{y}</b><br>" +
                         "Time: %{customdata[0]} - %{customdata[1]}<br>" +
                         "Shoplifting Prob: %{x:.1%}<br>" +
                         "Behavior: %{customdata[2]}<extra></extra>",
            customdata=[[e['start_time_str'], e['end_time_str'], e['behavior']] for e in events]
        ))

        # Add threshold line
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Threshold (50%)")

        fig.update_layout(
            title="Shoplifting Probability by Time Window",
            xaxis_title="Shoplifting Probability",
            yaxis_title="Time Window",
            height=max(300, len(events) * 30),
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Timeline visualization
        st.markdown("### Visual Timeline")

        # Create a timeline view
        fig2 = go.Figure()

        for i, e in enumerate(events):
            color = '#ff4b4b' if e['behavior'] == 'shoplifting' else '#00c853'
            fig2.add_trace(go.Scatter(
                x=[e['start_time'], e['end_time']],
                y=[0, 0],
                mode='lines',
                line=dict(color=color, width=20),
                name=e['behavior'],
                showlegend=i == 0 or events[i-1]['behavior'] != e['behavior'],
                hovertemplate=f"Window {e['window']}<br>" +
                             f"Time: {e['start_time_str']} - {e['end_time_str']}<br>" +
                             f"Behavior: {e['behavior']}<br>" +
                             f"Probability: {e['shoplifting_prob']:.1%}<extra></extra>"
            ))

        fig2.update_layout(
            title="Behavior Over Time",
            xaxis_title="Time (seconds)",
            height=150,
            showlegend=True,
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Detailed table
        with st.expander("View Detailed Analysis Table"):
            df_display = df_events[['window', 'start_time_str', 'end_time_str', 'behavior', 'shoplifting_prob', 'confidence']].copy()
            df_display.columns = ['Window', 'Start', 'End', 'Behavior', 'Shoplifting Prob', 'Confidence']
            df_display['Shoplifting Prob'] = df_display['Shoplifting Prob'].apply(lambda x: f"{x:.1%}")
            df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_display, use_container_width=True)

    # Video Info
    st.markdown("---")
    st.markdown("### Video Information")

    video_info = results['video_info']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Filename:** {video_info['filename']}")
    with col2:
        st.write(f"**Duration:** {video_info['duration']}")
    with col3:
        st.write(f"**Resolution:** {video_info['resolution']}")
    with col4:
        st.write(f"**FPS:** {video_info['fps']}")

    # Alert Summary
    st.markdown("---")
    st.markdown("### Advisory Summary")

    if risk == 'HIGH':
        st.markdown("""
        <div class='alert-high'>
            <h4 style='margin:0;'>⚠️ HIGH RISK ALERT</h4>
            <p style='margin:0.5rem 0;'>
                <strong>Action Required:</strong> Suspicious behavior patterns detected that are consistent
                with potential shoplifting activity. Immediate human review is strongly recommended.
            </p>
            <p style='margin:0; font-style:italic;'>
                This is an advisory system. Final determination requires human validation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif risk == 'MEDIUM':
        st.markdown("""
        <div class='alert-medium'>
            <h4 style='margin:0;'>⚡ MEDIUM RISK</h4>
            <p style='margin:0.5rem 0;'>
                <strong>Review Recommended:</strong> Some suspicious patterns have been detected.
                Human review is recommended to confirm or dismiss potential concerns.
            </p>
            <p style='margin:0; font-style:italic;'>
                This is an advisory system. Final determination requires human validation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif risk == 'LOW':
        st.markdown("""
        <div class='alert-low'>
            <h4 style='margin:0;'>ℹ️ LOW RISK</h4>
            <p style='margin:0.5rem 0;'>
                <strong>Minor Anomalies:</strong> Some slight deviations from normal behavior detected.
                Routine monitoring may be sufficient.
            </p>
            <p style='margin:0; font-style:italic;'>
                This is an advisory system. Final determination requires human validation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-none'>
            <h4 style='margin:0;'>✓ NORMAL BEHAVIOR</h4>
            <p style='margin:0.5rem 0;'>
                <strong>Status:</strong> Normal shopping behavior detected. No immediate concerns identified.
            </p>
            <p style='margin:0; font-style:italic;'>
                This is an advisory system. Periodic monitoring is still recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_pos_simulator():
    """Render the interactive POS simulator panel for real-time mode."""
    st.markdown("### POS Simulator")
    st.markdown("Scan items as they are purchased in real-time.")

    # Initialize POS session if not active
    if not st.session_state.pos_session_active:
        if st.button("Start New Transaction", type="primary", use_container_width=True):
            st.session_state.pos_simulator.start_session()
            st.session_state.pos_session_active = True
            st.session_state.pos_cart = []
            st.rerun()
        return None

    # Product buttons grid
    st.markdown("**Scan Items:**")
    catalog = get_product_catalog()

    # Display products in 2 columns
    cols = st.columns(2)
    for i, product in enumerate(catalog):
        col = cols[i % 2]
        with col:
            btn_label = f"{product['name']}\n${product['price']:.2f}"
            if st.button(btn_label, key=f"scan_{product['sku']}", use_container_width=True):
                # Scan the item
                scan_event = st.session_state.pos_simulator.scan_item(product['sku'])
                if scan_event:
                    st.session_state.pos_cart.append({
                        'sku': product['sku'],
                        'name': product['name'],
                        'price': product['price'],
                        'quantity': 1
                    })
                    st.rerun()

    st.markdown("---")

    # Cart display
    st.markdown("**Current Cart:**")
    if st.session_state.pos_cart:
        total = 0.0
        for item in st.session_state.pos_cart:
            st.write(f"- {item['name']}: ${item['price']:.2f}")
            total += item['price']
        st.markdown(f"**Total: ${total:.2f}**")
    else:
        st.info("No items scanned yet")

    # Transaction actions
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Complete Transaction", type="primary", use_container_width=True):
            transaction = st.session_state.pos_simulator.complete_transaction()
            st.session_state.pos_session_active = False
            if transaction:
                st.success(f"Transaction {transaction.transaction_id} completed!")
            st.rerun()

    with col2:
        if st.button("Cancel Transaction", use_container_width=True):
            st.session_state.pos_simulator.void_session()
            st.session_state.pos_session_active = False
            st.session_state.pos_cart = []
            st.rerun()

    # Return current POS data for analysis
    if st.session_state.pos_cart:
        return create_pos_data(st.session_state.pos_cart)
    return None


def render_pos_editor():
    """Render the POS data editor for upload mode."""
    st.markdown("### POS Transaction Editor")
    st.markdown("Configure the point-of-sale data to simulate different scenarios.")

    catalog = get_product_catalog()

    # Initialize session state for selected items
    if 'selected_items' not in st.session_state:
        st.session_state.selected_items = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Select Items for Transaction:**")

        # Create a multi-select for products
        selected_products = st.multiselect(
            "Choose products that were billed",
            options=[f"{p['sku']} - {p['name']} (${p['price']:.2f})" for p in catalog],
            default=[],
            help="Select items that appear in the POS transaction"
        )

        # Parse selected products
        selected_items = []
        for selection in selected_products:
            sku = selection.split(" - ")[0]
            product = next((p for p in catalog if p['sku'] == sku), None)
            if product:
                selected_items.append(product.copy())

        # Payment method
        payment_method = st.selectbox(
            "Payment Method",
            options=["card", "cash", "mobile"],
            index=0
        )

    with col2:
        st.markdown("**Transaction Preview:**")

        if selected_items:
            total = sum(item['price'] for item in selected_items)
            st.success(f"Items: {len(selected_items)}")
            st.info(f"Total: ${total:.2f}")

            with st.expander("View Items"):
                for item in selected_items:
                    st.write(f"- {item['name']}: ${item['price']:.2f}")
        else:
            st.warning("No items selected")
            st.caption("Select items to simulate a transaction, or leave empty to simulate no billing.")

    # Quick scenarios
    st.markdown("---")
    st.markdown("**Quick Scenarios:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Normal Purchase", help="3 items fully billed"):
            st.session_state.pos_scenario = "normal"
            return create_pos_data(catalog[:3], payment_method)

    with col2:
        if st.button("Partial Billing", help="1 item billed out of 3"):
            st.session_state.pos_scenario = "partial"
            return create_pos_data(catalog[:1], payment_method)

    with col3:
        if st.button("No Billing", help="No items billed"):
            st.session_state.pos_scenario = "none"
            return create_pos_data([], payment_method)

    # Return custom selection
    if selected_items:
        return create_pos_data(selected_items, payment_method)

    return None


# ============== MODE RENDERERS ==============

def render_realtime_mode():
    """Render the real-time camera mode UI."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Camera Connection")

        # RTSP URL input
        rtsp_url = st.text_input(
            "RTSP Camera URL",
            value=st.session_state.rtsp_url,
            placeholder="rtsp://username:password@192.168.1.100:554/stream",
            help="Enter your IP camera's RTSP URL"
        )
        st.session_state.rtsp_url = rtsp_url

        # Webcam fallback option
        use_webcam = st.checkbox(
            "Use webcam instead (for testing)",
            value=False,
            help="Use your computer's webcam if no RTSP camera available"
        )

        # Connection status
        st.markdown("---")

        if st.session_state.realtime_running:
            st.success("Camera: Connected")
            if st.button("Stop Analysis", type="secondary", use_container_width=True):
                st.session_state.realtime_running = False
                st.rerun()

            # Display real-time stats
            st.markdown("### Live Statistics")
            stats_placeholder = st.empty()

            # Simulated stats for demo (in real implementation, these come from RealtimeProcessor)
            with stats_placeholder.container():
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Frames", "Processing...")
                with stat_col2:
                    st.metric("Risk Score", "0.00")
                with stat_col3:
                    st.metric("Behaviors", "0")

            st.info("Real-time processing is running. Check Manager Dashboard for alerts.")

        else:
            st.warning("Camera: Not connected")

            can_start = bool(rtsp_url) or use_webcam
            if st.button(
                "Start Real-Time Analysis",
                type="primary",
                use_container_width=True,
                disabled=not can_start
            ):
                if can_start:
                    st.session_state.realtime_running = True
                    st.success("Starting real-time analysis...")
                    st.rerun()

            if not can_start:
                st.caption("Enter RTSP URL or enable webcam to start")

        # Webhook info
        st.markdown("---")
        st.markdown("### POS API Webhook")
        st.code("POST http://localhost:5001/webhook/pos", language="text")
        st.caption("External POS systems can send transactions to this endpoint")

    with col2:
        # POS Simulator
        st.markdown("### POS Simulator")
        pos_data = render_pos_simulator()
        if pos_data:
            st.session_state.pos_data = pos_data


def render_upload_mode():
    """Render the upload video mode UI (original functionality)."""
    # Video upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file to analyze",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for potential shoplifting behavior"
        )

        # Sample videos
        st.markdown("**Or select a sample video:**")

        sample_videos = []
        training_normal = Path("data/training/normal")
        training_shoplifting = Path("data/training/shoplifting")

        if training_normal.exists():
            normal_vids = list(training_normal.glob("*.mp4"))[:3]
            sample_videos.extend([(v, "normal") for v in normal_vids])
        if training_shoplifting.exists():
            shoplifting_vids = list(training_shoplifting.glob("*.mp4"))[:3]
            sample_videos.extend([(v, "shoplifting") for v in shoplifting_vids])

        selected_sample = None
        if sample_videos:
            options = ["-- Select a sample --"] + [f"{v.name} ({label})" for v, label in sample_videos]
            selected_option = st.selectbox("Sample videos", options, label_visibility="collapsed")
            if selected_option != "-- Select a sample --":
                idx = options.index(selected_option) - 1
                selected_sample = sample_videos[idx][0]

    with col2:
        # Video preview
        if uploaded_file:
            st.markdown("### Preview")
            st.video(uploaded_file)

    st.markdown("---")

    # POS Editor for upload mode
    pos_data = render_pos_editor()
    if pos_data:
        st.session_state.pos_data = pos_data

    st.markdown("---")

    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "Analyze Video",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None and selected_sample is None)
        )

    # Run analysis
    if analyze_clicked:
        video_path = None

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
        elif selected_sample is not None:
            video_path = str(selected_sample)

        if video_path:
            # Progress tracking
            st.markdown("### Processing")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)

            # Run analysis
            with st.spinner("Analyzing video..."):
                results = analyze_video(
                    video_path,
                    pos_data=st.session_state.pos_data,
                    progress_callback=update_progress
                )

            # Clean up temp file
            if uploaded_file is not None and os.path.exists(video_path):
                os.unlink(video_path)

            # Store results
            st.session_state.analysis_results = results

            # Clear progress
            progress_bar.empty()
            status_text.empty()


# ============== MAIN APP ==============

def main():
    """Main application entry point."""

    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'pos_data' not in st.session_state:
        st.session_state.pos_data = None

    # Initialize real-time mode session state
    init_realtime_session_state()

    # Render header and sidebar
    render_header()
    render_sidebar()

    # Check model status
    if not check_model_exists():
        st.error("Model not found. Please train the model first.")
        st.code("python train.py", language="bash")

        st.markdown("---")
        st.markdown("### While you wait...")
        st.info("""
        The model needs to be trained before analysis can run.
        Training uses videos from:
        - `data/training/normal/` - Normal shopping behavior
        - `data/training/shoplifting/` - Shoplifting behavior

        Run the training command above to get started.
        """)
        return

    # Main tabs
    tab1, tab2 = st.tabs(["📊 Model Performance", "🎥 Video Analysis"])

    # Tab 1: Model Performance
    with tab1:
        render_model_performance_tab()

    # Tab 2: Video Analysis
    with tab2:
        st.markdown('<div class="section-header">Video Analysis</div>', unsafe_allow_html=True)

        # Mode selector
        st.markdown("### Select Analysis Mode")
        mode_col1, mode_col2 = st.columns(2)

        with mode_col1:
            if st.button(
                "📹 Real-Time Camera",
                type="primary" if st.session_state.video_mode == "realtime" else "secondary",
                use_container_width=True,
                help="Connect to RTSP camera for live analysis"
            ):
                st.session_state.video_mode = "realtime"
                st.rerun()

        with mode_col2:
            if st.button(
                "📁 Upload Video",
                type="primary" if st.session_state.video_mode == "upload" else "secondary",
                use_container_width=True,
                help="Upload pre-recorded video for analysis"
            ):
                st.session_state.video_mode = "upload"
                st.rerun()

        st.markdown("---")

        # =====================================================================
        # REAL-TIME MODE
        # =====================================================================
        if st.session_state.video_mode == "realtime":
            render_realtime_mode()

        # =====================================================================
        # UPLOAD MODE (Original functionality)
        # =====================================================================
        else:
            render_upload_mode()

        # Display results (shared between modes)
        if st.session_state.analysis_results:
            st.markdown("---")
            render_analysis_results(st.session_state.analysis_results)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem 0;'>
        <p style='font-size: 1.1rem; font-weight: 500;'>Digital Witness - Retail Security Assistant</p>
        <p style='font-size: 0.9rem;'>
            This is an advisory system. All alerts require human validation.<br>
            The system does NOT determine guilt - it provides evidence for human review.
        </p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>Final Year Project - IIT 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
