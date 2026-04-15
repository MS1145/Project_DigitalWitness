"""
ui/components.py - Stateless header and sidebar renderers.
"""
from __future__ import annotations
import streamlit as st


def render_header() -> None:
    st.markdown('<h1 class="main-title">Digital Witness</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Deep Learning Retail Security Assistant '
        '| YOLO26 + MobileNetV2 + LSTM</p>',
        unsafe_allow_html=True,
    )


def render_sidebar(system_status: dict[str, bool]) -> None:
    """
    Args:
        system_status : dict from ModelRegistry.system_status()
    """
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("### About Digital Witness")
        st.markdown("""
        **Digital Witness** detects potential shoplifting from surveillance video
        using deep learning.

        **Pipeline:**
        1. **YOLO26 Detection** - Jan 2026 model, NMS-free, 43% faster than YOLOv8.
           Identifies and tracks people & products.
        2. **MobileNetV2 Features** - Extracts 1280-dim spatial features per frame
           (fine-tuned backbone, ImageNet pretrained).
        3. **Bidirectional LSTM + Attention** - Classifies temporal sequences of
           features into normal / shoplifting.

        **Important:** This is an *advisory system*. It does **not** determine guilt.
        All alerts require human validation.
        """)
        st.markdown("---")
        st.markdown("### System Status")
        if system_status.get("all_ready"):
            st.success("All models loaded")
        else:
            st.warning("Some models missing - check `models/` folder")
        st.markdown("---")
        st.markdown("### Important Notice")
        st.info(
            "This is an **advisory system**.\n"
            "All alerts require human validation.\n"
            "The system does NOT determine guilt."
        )
        st.markdown("---")
        st.markdown("##### Final Year Project - 2026")
