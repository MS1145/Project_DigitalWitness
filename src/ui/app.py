"""
Digital Witness - Deep Learning Web Interface

A Streamlit-based web interface for the YOLO -> CNN -> LSTM behavior detection system.
Features:
- Video upload and deep learning analysis
- Real-time processing progress
- Analysis results with visualizations
- Bias and fairness metrics display

"""
import streamlit as st
import tempfile
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path (go up from src/ui/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Digital Witness - Retail Security Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styles */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-none {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1e3a5f;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 3px solid #3d7ab5;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }

    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    /* Card container */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============== HELPER FUNCTIONS ==============

def check_model_exists():
    """Check if the trained LSTM model exists."""
    from src.config import MODELS_DIR
    lstm_path = MODELS_DIR / "lstm_classifier.pt"
    return lstm_path.exists()


def load_model_metrics():
    """Load deep learning model performance metrics."""
    from src.config import MODELS_DIR
    info_path = MODELS_DIR / "lstm_classifier_info.json"

    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return None


def get_product_catalog():
    """Get the product catalog for POS data."""
    return [
        {"sku": "ITEM001", "name": "Snack Bar", "price": 2.99},
        {"sku": "ITEM002", "name": "Soda Bottle", "price": 1.99},
        {"sku": "ITEM003", "name": "Chocolate Box", "price": 5.99},
        {"sku": "ITEM004", "name": "Energy Drink", "price": 3.49},
        {"sku": "ITEM005", "name": "Chips Bag", "price": 2.49},
        {"sku": "ITEM006", "name": "Candy Pack", "price": 1.49},
        {"sku": "ITEM007", "name": "Gum Pack", "price": 0.99},
        {"sku": "ITEM008", "name": "Protein Bar", "price": 3.99},
        {"sku": "ITEM009", "name": "Water Bottle", "price": 1.29},
        {"sku": "ITEM010", "name": "Coffee Can", "price": 2.79},
    ]


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


def analyze_video_deep(video_path, progress_callback=None):
    """
    Analyze a video using the deep learning pipeline (YOLO -> CNN -> LSTM).
    """
    from src.main import run_pipeline

    # Run the deep learning pipeline
    results = run_pipeline(
        video_path=Path(video_path),
        progress_callback=progress_callback
    )

    return results


# ============== UI COMPONENTS ==============

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-title">Digital Witness</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning Retail Security Assistant | YOLO + CNN + LSTM</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("### About Digital Witness")
        st.markdown("""
        **Digital Witness** is an intelligent retail security assistant that uses deep learning to detect potential shoplifting behavior from surveillance video footage.

        **How It Works:**
        1. **YOLO Detection** - Identifies and tracks people and products in real-time
        2. **CNN Features** - Extracts visual patterns from each frame using ResNet18
        3. **LSTM Classification** - Analyzes temporal behavior sequences to detect suspicious patterns

        **Key Features:**
        - Real-time behavior analysis
        - Explainable risk scoring
        - Human-in-the-loop design
        - Privacy-conscious approach

        **Important:** This is an *advisory system* that assists human operators. It does NOT determine guilt - all alerts require human validation.
        """)

        st.markdown("---")

        # Model Status
        st.markdown("### System Status")
        if check_model_exists():
            st.success("LSTM Model: Ready")
            metrics = load_model_metrics()
            if metrics:
                # Use final_val_acc from training, fallback to metrics.accuracy
                acc = metrics.get('final_val_acc', metrics.get('metrics', {}).get('accuracy', 0))
                st.metric("Model Accuracy", f"{acc:.1%}")
        else:
            st.warning("LSTM Model: Not trained")
            st.info("Run `python run.py --train`")

        st.markdown("---")

        st.markdown("### Pipeline")
        st.code("YOLO -> CNN -> LSTM", language=None)

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
        st.warning("No model metrics found. Please train the model first using `python run.py --train`")

        st.markdown("---")
        st.markdown("### Training Instructions")
        st.markdown("""
        To train the deep learning model:

        1. Place training videos in:
           - `data/training/normal/` - Normal shopping behavior
           - `data/training/shoplifting/` - Shoplifting behavior

        2. Run the training command:
        """)
        st.code("python run.py --train", language="bash")

        st.markdown("""
        The training pipeline will:
        - Extract CNN features from video frames
        - Train a bidirectional LSTM classifier
        - Save the model to `models/lstm_classifier.pt`
        """)
        return

    st.markdown('<div class="section-header">Deep Learning Model Performance</div>', unsafe_allow_html=True)

    # Training info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Date", metrics.get('training_date', 'N/A')[:10] if metrics.get('training_date') else 'N/A')
    with col2:
        st.metric("Training Samples", f"{metrics.get('n_train_samples', 0):,}")
    with col3:
        st.metric("Validation Samples", f"{metrics.get('n_val_samples', metrics.get('n_test_samples', 0)):,}")

    st.markdown("---")

    # Main metrics
    st.markdown("### Classification Metrics")

    # Check if detailed metrics are available
    detailed = metrics.get('metrics', {})
    if detailed:
        # Show detailed metrics (precision, recall, F1)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{detailed.get('accuracy', metrics.get('final_val_acc', 0)):.1%}", help="Overall correct predictions")
        with col2:
            st.metric("Precision", f"{detailed.get('precision', 0):.1%}", help="True positives / All predicted positives")
        with col3:
            st.metric("Recall", f"{detailed.get('recall', 0):.1%}", help="True positives / All actual positives")
        with col4:
            st.metric("F1 Score", f"{detailed.get('f1_score', 0):.1%}", help="Harmonic mean of precision and recall")
    else:
        # Fallback to basic accuracy
        col1, col2, col3, col4 = st.columns(4)

        train_acc = metrics.get('final_train_acc', 0)
        val_acc = metrics.get('final_val_acc', 0)

        with col1:
            st.metric("Training Accuracy", f"{train_acc:.1%}", help="Accuracy on training data")
        with col2:
            st.metric("Validation Accuracy", f"{val_acc:.1%}", help="Accuracy on validation data")
        with col3:
            st.metric("Normal Videos", metrics.get('n_normal_videos', 0), help="Number of normal training videos")
        with col4:
            st.metric("Shoplifting Videos", metrics.get('n_shoplifting_videos', 0), help="Number of shoplifting training videos")

        st.info("Run `python run.py --evaluate` to compute detailed metrics (Precision, Recall, F1)")

    st.markdown("---")

    # Architecture info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
        **Detection Layer (YOLO v8)**
        - Real-time object detection
        - Multi-object tracking
        - Person & product identification

        **Feature Extraction (CNN - ResNet18)**
        - Pre-trained ImageNet weights
        - 512-dimensional feature vectors
        - Spatial pattern recognition

        **Temporal Classification (Bidirectional LSTM)**
        - 256 hidden units
        - 2 LSTM layers
        - Attention mechanism
        - Binary classification output
        """)

    with col2:
        st.markdown("### Confusion Matrix")

        import plotly.graph_objects as go

        # Check if confusion matrix is available
        if 'confusion_matrix' in metrics:
            conf_matrix = np.array(metrics['confusion_matrix'])
            classes = metrics.get('classes', ['normal', 'shoplifting'])

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
        else:
            # Fallback: show training data distribution
            classes = metrics.get('classes', ['normal', 'shoplifting'])
            n_normal = metrics.get('n_normal_videos', 0)
            n_shoplifting = metrics.get('n_shoplifting_videos', 0)

            fig = go.Figure(data=[
                go.Bar(
                    x=classes,
                    y=[n_normal, n_shoplifting],
                    marker_color=['#38ef7d', '#ff5722']
                )
            ])
            fig.update_layout(
                title="Training Data Distribution",
                xaxis_title="Class",
                yaxis_title="Number of Videos",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("Run `python run.py --evaluate` to generate confusion matrix")

    # Per-class metrics if available
    per_class = metrics.get('per_class_metrics', {})
    if per_class:
        st.markdown("---")
        st.markdown("### Per-Class Metrics")

        classes = metrics.get('classes', ['normal', 'shoplifting'])
        precision_dict = per_class.get('precision', {})
        recall_dict = per_class.get('recall', {})
        f1_dict = per_class.get('f1', {})

        # Create a table
        metrics_data = []
        for cls in classes:
            metrics_data.append({
                "Class": cls.capitalize(),
                "Precision": f"{precision_dict.get(cls, 0):.1%}",
                "Recall": f"{recall_dict.get(cls, 0):.1%}",
                "F1 Score": f"{f1_dict.get(cls, 0):.1%}"
            })

        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Training Configuration
    st.markdown("---")
    st.markdown("### Training Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**LSTM Parameters**")
        lstm_params = {
            'input_dim': metrics.get('input_dim', 512),
            'hidden_size': 256,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'sequence_length': metrics.get('sequence_length', 30)
        }
        st.json(lstm_params)
    with col2:
        st.markdown("**CNN Backbone**")
        st.json({
            'model': 'ResNet18',
            'pretrained': True,
            'feature_dim': metrics.get('input_dim', 512),
            'freeze_layers': True
        })
    with col3:
        st.markdown("**Training Settings**")
        training_config = {
            'epochs': metrics.get('epochs', 50),
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'stride': metrics.get('stride', 15)
        }
        st.json(training_config)


def render_analysis_results(results):
    """Render the deep learning analysis results."""
    if not results.get('success', False):
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    import plotly.graph_objects as go
    import plotly.express as px

    # Check if running in demo mode
    if results.get('demo_mode', False):
        st.warning("**DEMO MODE**: Analysis ran with simulated data. The actual video analysis failed or no video was provided. Results shown are for demonstration purposes only.")
        st.markdown("---")

    # Main prediction banner
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    # Get key results
    intent_score = results.get('intent_score', {})
    alert = results.get('alert')
    quality = results.get('quality_analysis', {})
    bias_report = results.get('bias_report', {})

    score = intent_score.get('score', 0)
    severity = intent_score.get('severity', 'NONE').upper()  # Normalize to uppercase

    # Result banner based on severity
    if severity == 'CRITICAL':
        st.markdown(f"""
        <div class='alert-critical'>
            <h2 style='margin:0; font-size:1.8rem;'>CRITICAL ALERT</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Intent Score: {score:.2f} | Severity: {severity.upper()}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif severity == 'HIGH':
        st.markdown(f"""
        <div class='alert-high'>
            <h2 style='margin:0; font-size:1.8rem;'>HIGH RISK DETECTED</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Intent Score: {score:.2f} | Severity: {severity}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif severity == 'MEDIUM':
        st.markdown(f"""
        <div class='alert-medium'>
            <h2 style='margin:0; font-size:1.8rem;'>MEDIUM RISK</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Intent Score: {score:.2f} | Severity: {severity}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif severity == 'LOW':
        st.markdown(f"""
        <div class='alert-low'>
            <h2 style='margin:0; font-size:1.8rem;'>LOW RISK</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Intent Score: {score:.2f} | Severity: {severity}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:  # NONE or unknown
        st.markdown(f"""
        <div class='alert-none'>
            <h2 style='margin:0; font-size:1.8rem;'>NORMAL BEHAVIOR</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>
                Intent Score: {score:.2f} | No concerns detected
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Detection Statistics
    st.markdown("### Detection Statistics")

    detections = results.get('detections', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Persons Tracked", detections.get('persons_tracked', 0))
    with col2:
        st.metric("Products Detected", detections.get('products_detected', 0))
    with col3:
        st.metric("Interactions Found", detections.get('interactions', 0))
    with col4:
        st.metric("Frames Processed", detections.get('frames_processed', 0))

    st.markdown("---")

    # Intent Score Breakdown
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Intent Score")

        # Intent gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
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
        st.markdown("### Score Components")

        components = intent_score.get('components', {})
        if components:
            # Create component breakdown chart
            comp_names = list(components.keys())
            comp_values = list(components.values())

            fig = go.Figure(go.Bar(
                x=comp_values,
                y=comp_names,
                orientation='h',
                marker_color='#667eea'
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(range=[0, 1], tickformat='.0%')
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Explanation:**")
        st.info(intent_score.get('explanation', 'No explanation available'))

    st.markdown("---")

    # Behavior Timeline
    st.markdown("### Behavior Timeline")

    behavior_events = results.get('behavior_events', [])
    if behavior_events:
        # Create timeline data
        events_data = []
        for event in behavior_events:
            events_data.append({
                'Behavior': event.get('behavior_type', 'unknown'),
                'Start': event.get('start_time', 0),
                'End': event.get('end_time', 0),
                'Confidence': event.get('confidence', 0)
            })

        df_events = pd.DataFrame(events_data)

        if not df_events.empty:
            # Timeline visualization
            fig = go.Figure()

            colors = {
                'normal': '#00c853',
                'pickup': '#ffc107',
                'concealment': '#ff5722',
                'bypass': '#f44336',
                'shoplifting': '#e91e63'
            }

            for i, row in df_events.iterrows():
                color = colors.get(row['Behavior'], '#667eea')
                fig.add_trace(go.Scatter(
                    x=[row['Start'], row['End']],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color=color, width=25),
                    name=row['Behavior'],
                    showlegend=i == 0 or (i > 0 and df_events.iloc[i-1]['Behavior'] != row['Behavior']),
                    hovertemplate=f"Behavior: {row['Behavior']}<br>" +
                                 f"Time: {row['Start']:.1f}s - {row['End']:.1f}s<br>" +
                                 f"Confidence: {row['Confidence']:.1%}<extra></extra>"
                ))

            fig.update_layout(
                title="Behavior Over Time",
                xaxis_title="Time (seconds)",
                height=150,
                showlegend=True,
                yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            with st.expander("View Detailed Behavior Events"):
                df_display = df_events.copy()
                df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.1%}")
                df_display['Start'] = df_display['Start'].apply(lambda x: f"{x:.1f}s")
                df_display['End'] = df_display['End'].apply(lambda x: f"{x:.1f}s")
                st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No behavior events detected in the video")

    st.markdown("---")

    # Quality Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Video Quality Analysis")

        reliability = quality.get('reliability_score', 0)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = reliability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Reliability Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#38ef7d"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#fff3cd"},
                    {'range': [75, 100], 'color': "#d4edda"}
                ]
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        quality_issues = quality.get('issues', [])
        if quality_issues:
            st.warning("Quality Issues Detected:")
            for issue in quality_issues:
                st.write(f"- {issue}")

    with col2:
        st.markdown("### Bias & Fairness Metrics")

        if bias_report:
            fairness_score = bias_report.get('overall_fairness_score', 0)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fairness_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fairness Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 75], 'color': "#fff3cd"},
                        {'range': [75, 100], 'color': "#d4edda"}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            bias_flags = bias_report.get('flags', [])
            if bias_flags:
                st.warning("Bias Concerns:")
                for flag in bias_flags:
                    st.write(f"- {flag}")
            else:
                st.success("No significant bias detected")
        else:
            st.info("Bias analysis not available")

    # Video Info
    st.markdown("---")
    st.markdown("### Video Information")

    video_metadata = results.get('video_metadata', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Filename:** {video_metadata.get('filename', 'N/A')}")
    with col2:
        st.write(f"**Duration:** {video_metadata.get('duration', 0):.1f}s")
    with col3:
        st.write(f"**Resolution:** {video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}")
    with col4:
        st.write(f"**FPS:** {video_metadata.get('fps', 0)}")

    # Alert Summary
    st.markdown("---")
    st.markdown("### Advisory Summary")

    if alert:
        alert_level = alert.get('level', 'NONE').upper()
        if alert_level in ['CRITICAL', 'HIGH']:
            st.markdown(f"""
            <div class='alert-high'>
                <h4 style='margin:0;'>Alert Generated: {alert.get('alert_id', 'N/A')}</h4>
                <p style='margin:0.5rem 0;'>
                    <strong>Action Required:</strong> {alert.get('message', 'Review recommended')}
                </p>
                <p style='margin:0; font-style:italic;'>
                    This is an advisory system. Final determination requires human validation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-low'>
                <h4 style='margin:0;'>Alert ID: {alert.get('alert_id', 'N/A')}</h4>
                <p style='margin:0.5rem 0;'>
                    {alert.get('message', 'No immediate concerns')}
                </p>
                <p style='margin:0; font-style:italic;'>
                    This is an advisory system. Periodic monitoring recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-none'>
            <h4 style='margin:0;'>No Alert Generated</h4>
            <p style='margin:0.5rem 0;'>
                <strong>Status:</strong> Normal shopping behavior detected. No immediate concerns identified.
            </p>
            <p style='margin:0; font-style:italic;'>
                This is an advisory system. Periodic monitoring is still recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_pos_editor():
    """Render the POS data editor."""
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


# ============== MAIN APP ==============

def main():
    """Main application entry point."""

    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Render header and sidebar
    render_header()
    render_sidebar()

    # Main tabs
    tab1, tab2 = st.tabs(["📊 Model Performance", "🎥 Video Analysis"])

    # Tab 1: Model Performance
    with tab1:
        render_model_performance_tab()

    # Tab 2: Video Analysis
    with tab2:
        st.markdown('<div class="section-header">Deep Learning Video Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>Pipeline:</strong> YOLO (detection) → CNN (features) → LSTM (classification)
        </div>
        """, unsafe_allow_html=True)

        # Video upload section
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file to analyze",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file to analyze for potential shoplifting behavior"
            )

        with col2:
            # Video preview
            if uploaded_file:
                st.markdown("### Preview")
                st.video(uploaded_file)

        st.markdown("---")

        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_clicked = st.button(
                "Analyze Video (YOLO + CNN + LSTM)",
                type="primary",
                use_container_width=True,
                disabled=(uploaded_file is None)
            )

        # Run analysis
        if analyze_clicked:
            video_path = None

            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name

            if video_path:
                # Progress tracking
                st.markdown("### Processing")
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, message):
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(message)

                # Run analysis
                with st.spinner("Running deep learning analysis..."):
                    try:
                        results = analyze_video_deep(
                            video_path,
                            progress_callback=update_progress
                        )
                    except Exception as e:
                        results = {
                            'success': False,
                            'error': str(e)
                        }

                # Clean up temp files
                if uploaded_file is not None and os.path.exists(video_path):
                    os.unlink(video_path)

                # Store results
                st.session_state.analysis_results = results

                # Clear progress
                progress_bar.empty()
                status_text.empty()

        # Display results
        if st.session_state.analysis_results:
            st.markdown("---")
            render_analysis_results(st.session_state.analysis_results)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem 0;'>
        <p style='font-size: 1.1rem; font-weight: 500;'>Digital Witness - Deep Learning Retail Security Assistant</p>
        <p style='font-size: 0.9rem;'>
            Pipeline: YOLO (detection) → CNN (features) → LSTM (classification)<br>
            This is an advisory system. All alerts require human validation.
        </p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>Final Year Project - IIT 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
