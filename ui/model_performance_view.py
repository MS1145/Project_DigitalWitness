"""
ui/model_performance_view.py - Renders the Model Performance tab.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.model_registry import ModelRegistry


class ModelPerformanceView:
    """Renders BiLSTM and MobileNetV2 training metrics from saved JSON info files."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def render(self) -> None:
        bilstm_info    = self._registry.load_bilstm_info()
        mobilenet_info = self._registry.load_mobilenet_info()

        if not bilstm_info and not mobilenet_info:
            st.warning("No model metrics found. Train the models first using the notebook.")
            return

        st.markdown('<div class="section-header">Deep Learning Model Performance</div>',
                    unsafe_allow_html=True)

        self._render_mobilenet_metrics(mobilenet_info or {})
        st.markdown("---")
        self._render_bilstm_metrics(bilstm_info or {})
        st.markdown("---")
        self._render_confusion_matrix(bilstm_info or {})
        self._render_per_class_table(bilstm_info or {})
        st.markdown("---")
        self._render_architecture_section(bilstm_info or {})

    #  Private section renderers ─

    def _render_mobilenet_metrics(self, mn: dict) -> None:
        mn_m = mn.get("metrics", {})
        st.markdown("### MobileNetV2 Frame Classifier - Evaluation Results")
        st.caption(
            f"Evaluated on {mn.get('n_val_samples', 0):,} held-out validation frames "
            "(20% stratified split)"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{mn_m.get('accuracy',  mn.get('best_val_acc', 0)):.2%}")
        c2.metric("Precision", f"{mn_m.get('precision', 0):.2%}")
        c3.metric("Recall",    f"{mn_m.get('recall',    0):.2%}")
        c4.metric("F1 Score",  f"{mn_m.get('f1_score',  0):.2%}")

        per_class = mn.get("per_class_metrics", {})
        if per_class:
            st.markdown("#### Per-Class Metrics")
            classes = mn.get("classes", ["normal", "shoplifting"])
            rows = [{"Class"    : c.capitalize(),
                     "Precision": f"{per_class.get('precision', {}).get(c, 0):.2%}",
                     "Recall"   : f"{per_class.get('recall',    {}).get(c, 0):.2%}",
                     "F1 Score" : f"{per_class.get('f1',        {}).get(c, 0):.2%}",
                     "Support"  : per_class.get('support', {}).get(c, 0)}
                    for c in classes]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def _render_bilstm_metrics(self, bi: dict) -> None:
        st.markdown("### BiLSTM + Attention Sequence Classifier - Validation Accuracy")
        st.caption("Best validation accuracy recorded during training (frame-sequence level)")
        if bi:
            c1, c2 = st.columns(2)
            c1.metric("Best Validation Accuracy", f"{bi.get('best_val_acc', 0):.2%}")
            c2.metric("Val Sequences", bi.get("n_val_samples", "N/A"))

    def _render_confusion_matrix(self, bi: dict) -> None:
        paths = self._registry.paths
        if paths.confusion_matrix.exists() and paths.learning_curve.exists():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Confusion Matrix")
                st.image(str(paths.confusion_matrix), use_container_width=True)
            with col2:
                st.markdown("### Training Learning Curve")
                st.image(str(paths.learning_curve), use_container_width=True)
        elif paths.confusion_matrix.exists():
            st.markdown("### Confusion Matrix")
            st.image(str(paths.confusion_matrix), use_container_width=True)
        elif paths.learning_curve.exists():
            st.markdown("### Training Learning Curve")
            st.image(str(paths.learning_curve), use_container_width=True)
        elif "confusion_matrix" in bi:
            st.markdown("### Confusion Matrix")
            cm      = np.array(bi["confusion_matrix"])
            classes = bi.get("classes", ["normal", "shoplifting"])
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=classes, y=classes, colorscale="Blues",
                text=cm, texttemplate="%{text}", textfont={"size": 20},
            ))
            fig.update_layout(title="Predicted vs Actual",
                               xaxis_title="Predicted", yaxis_title="True",
                               height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    def _render_per_class_table(self, bi: dict) -> None:
        per_class = bi.get("per_class_metrics", {})
        if not per_class:
            return
        st.markdown("---")
        st.markdown("### Per-Class Metrics")
        classes = bi.get("classes", ["normal", "shoplifting"])
        rows = [{"Class"    : c.capitalize(),
                 "Precision": f"{per_class.get('precision', {}).get(c, 0):.1%}",
                 "Recall"   : f"{per_class.get('recall',    {}).get(c, 0):.1%}",
                 "F1 Score" : f"{per_class.get('f1',        {}).get(c, 0):.1%}"}
                for c in classes]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def _render_architecture_section(self, bi: dict) -> None:
        st.markdown("### Model Architecture")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Detection: YOLO26** (Jan 2026)
            - NMS-free, 43% faster than YOLOv8
            - Fine-tuned on retail product annotations
            - ByteTrack multi-object tracking

            **Feature Extraction: MobileNetV2**
            - ImageNet pretrained, last 3 blocks fine-tuned
            - 1280-dim feature output (no projection layer)
            """)
        with col2:
            st.markdown("""
            **Temporal Classification: Bidirectional LSTM**
            - 256 hidden units, 2 layers, attention mechanism
            - Sequence length: 45 frames, stride: 15
            - 2-class output: normal / shoplifting

            **XAI Layer**
            - Attention weights → temporal explanation
            - Intent score: peak confidence × sustained factor
            - Bias adjustment via model availability check
            """)

        st.markdown("---")
        st.markdown("### Training Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**BiLSTM**")
            st.json({"input_dim"    : bi.get("input_dim", 1280),
                     "hidden"       : bi.get("hidden", 256),
                     "layers"       : bi.get("layers", 2),
                     "bidirectional": True,
                     "dropout"      : 0.3,
                     "seq_length"   : bi.get("seq_len", 45)})
        with c2:
            st.markdown("**MobileNetV2**")
            st.json({"backbone"       : "mobilenet_v2",
                     "weights"        : "MobileNet_V2_Weights.DEFAULT",
                     "feature_dim"    : 1280,
                     "unfrozen_blocks": "features[16:]"})
        with c3:
            st.markdown("**Training**")
            st.json({"batch_size"               : 16,
                     "lr_mobilenet"             : "1e-4",
                     "lr_bilstm"                : "1e-3",
                     "optimizer"                : "Adam",
                     "scheduler"                : "ReduceLROnPlateau",
                     "early_stopping_patience"  : 10})
