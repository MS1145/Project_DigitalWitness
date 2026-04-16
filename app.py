"""
app.py - Digital Witness entry point.

session state, tab layout, pipeline dispatch.
All ML logic lives in core/ and all rendering logic lives in ui/.

Streamlit is the web framework used for the entire front-end:
    Streamlit Inc. (2019). Streamlit - The fastest way to build data apps.
    https://streamlit.io
    Apache 2.0 License.

IMPORTANT: st.set_page_config() must be the very first Streamlit call in
the script. Streamlit throws an error if any other st.* call comes before it.
This is a known Streamlit constraint - do not reorder the imports below.
"""
import streamlit as st

# page config MUST be the very first Streamlit call
st.set_page_config(
    page_title="Digital Witness - Retail Security Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# remaining imports (after set_page_config)
import tempfile
import os
from pathlib import Path

from core.config import DEFAULT_PATHS, DEFAULT_PARAMS
from core.model_registry import ModelRegistry
from core.pipeline import AnalysisPipeline
from core.clip_extractor import ClipExtractor
from core.result_types import PipelineResult

from ui.styles import CSS_BLOCK
from ui.components import render_header, render_sidebar
from ui.analysis_view import AnalysisView
from ui.pos_audit_view import PosAuditView


class DigitalWitnessApp:

    def __init__(self) -> None:
        self._registry     = ModelRegistry(DEFAULT_PATHS)
        self._pipeline     = AnalysisPipeline(self._registry, DEFAULT_PARAMS)
        self._clip_extractor = ClipExtractor()
        self._analysis_view  = AnalysisView()
        self._pos_view       = PosAuditView()

    # public entry point
    def run(self) -> None:
        st.markdown(CSS_BLOCK, unsafe_allow_html=True)
        self._init_session_state()
        render_header()
        render_sidebar(self._registry.system_status())
        self._render_tabs()
        self._render_footer()

    # session state
    def _init_session_state(self) -> None:
        # only initialise keys that don't exist yet - this preserves values across reruns
        defaults = {
            "analysis_result"   : None,
            "pos_items"         : [],
            "last_uploaded_name": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # tab layout
    def _render_tabs(self) -> None:
        tab_video, tab_pos = st.tabs(["Video Analysis", "POS Audit"])
        with tab_video:
            self._render_video_tab()
        with tab_pos:
            self._pos_view.render(st.session_state.analysis_result)

    # video analysis tab
    def _render_video_tab(self) -> None:
        st.markdown(
            '<div class="section-header">Video Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-box">
                Upload a surveillance video clip. The pipeline runs YOLO26 detection,
                MobileNetV2 feature extraction, and BiLSTM sequence classification
                to identify potential shoplifting behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload surveillance video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported: MP4, AVI, MOV, MKV",
        )

        # clear stale result when a new file is uploaded so we don't show
        # last video's results under the new video
        if uploaded is not None:
            if st.session_state.last_uploaded_name != uploaded.name:
                st.session_state.analysis_result  = None
                st.session_state.last_uploaded_name = uploaded.name

        col_btn, col_clear = st.columns([1, 5])
        with col_btn:
            run_btn = st.button("Analyse Video", use_container_width=True)
        with col_clear:
            if st.button("Clear Results", key="clear_results"):
                st.session_state.analysis_result   = None
                st.session_state.last_uploaded_name = None
                st.rerun()

        if run_btn and uploaded is not None:
            self._run_analysis(uploaded)
        elif run_btn and uploaded is None:
            st.warning("Please upload a video file first.")

        if st.session_state.analysis_result is not None:
            self._render_analysis_output()

    # pipeline dispatch
    def _run_analysis(self, uploaded) -> None:
        import uuid
        suffix = Path(uploaded.name).suffix or ".mp4"
        # write the uploaded file to a temp path so OpenCV can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        visual_out = os.path.join(
            tempfile.gettempdir(),
            f"dw_annotated_{uuid.uuid4().hex[:8]}.mp4",
        )

        # live preview layout (columns created once; only st.empty() placeholders update per-frame)
        # creating columns inside the callback would rebuild the layout every frame and flicker
        st.markdown("### Live Analysis Preview")
        st.caption(
            "Watch the pipeline analyse your video - YOLO detections with "
            "coloured bounding boxes, classification badge and stats update in real time."
        )
        _col_frame, _col_stats = st.columns([3, 1])
        with _col_frame:
            _frame_ph = st.empty()
            _frame_ph.markdown(
                "<div style='height:200px;display:flex;align-items:center;"
                "justify-content:center;background:#111;border-radius:8px;"
                "color:#666;font-size:0.9rem;'>Waiting for first frame...</div>",
                unsafe_allow_html=True,
            )
        with _col_stats:
            _stat_ph = st.empty()

        _timeline_ph = st.empty()
        _prog        = st.progress(0)
        _status      = st.empty()

        def on_progress(pct: float, msg: str) -> None:
            _prog.progress(min(int(pct), 100))
            _status.text(msg)

        def on_live_frame(frame_rgb, frame_num: int, total: int, stats: dict) -> None:
            # left panel - annotated video frame
            _frame_ph.image(
                frame_rgb,
                caption=f"Frame {frame_num} / {total}",
                use_container_width=True,
            )

            # right panel - dark stats panel with current classification
            label    = stats.get("label", "Analyzing...")
            conf     = stats.get("conf", 0.0)
            persons  = stats.get("persons_tracked", 0)
            products = stats.get("products_detected", 0)
            # colour the label based on verdict - red for shoplifting, green for normal
            col = ("#dc3545" if label == "shoplifting" else
                   "#28a745" if label == "normal" else "#aaaaaa")
            _stat_ph.markdown(f"""
            <div style='padding:1rem;background:#1a1a2e;border-radius:8px;
                        border:1px solid #333;font-family:monospace;'>
                <p style='color:#888;margin:0;font-size:0.72rem;'>CLASSIFICATION</p>
                <p style='color:{col};font-size:1.15rem;font-weight:700;
                           margin:0.25rem 0;'>{label.upper()}</p>
                <p style='color:#ccc;font-size:0.85rem;margin:0;'>
                    {conf:.0%} confidence</p>
                <hr style='border-color:#333;margin:0.6rem 0;'>
                <p style='color:#888;margin:0;font-size:0.72rem;'>
                    PEOPLE INTERACTED WITH THE PRODUCTS</p>
                <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>
                    {persons}</p>
                <hr style='border-color:#333;margin:0.6rem 0;'>
                <p style='color:#888;margin:0;font-size:0.72rem;'>PICKED UP PRODUCTS</p>
                <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>
                    {products}</p>
            </div>""", unsafe_allow_html=True)

            # bottom - colour timeline bar showing where we are in the video
            timeline = stats.get("timeline", [])
            if timeline:
                pct  = frame_num / max(total, 1)
                cmap = {"red": "#dc3545", "green": "#28a745", "gray": "#555"}
                bar  = ('<div style="display:flex;width:100%;height:10px;'
                        'border-radius:4px;overflow:hidden;background:#333;">')
                for tc in timeline:
                    w    = 100.0 / max(len(timeline), 1)
                    bar += (f'<div style="flex:{w:.2f};'
                            f'background:{cmap.get(tc, "#555")};"></div>')
                bar += "</div>"
                _timeline_ph.markdown(f"""
                <div style='margin:0.4rem 0 0.8rem 0;'>
                    <p style='color:#888;font-size:0.72rem;margin-bottom:4px;'>
                        Timeline &nbsp;
                        <span style='color:#28a745;'>Normal</span>&nbsp;
                        <span style='color:#dc3545;'>Suspicious</span>&nbsp;
                        <span style='color:#555;'>Analyzing</span>
                    </p>
                    {bar}
                    <p style='color:#666;font-size:0.72rem;margin-top:4px;'>
                        {pct:.0%} processed</p>
                </div>""", unsafe_allow_html=True)

        try:
            result: PipelineResult = self._pipeline.run(
                video_path=tmp_path,
                progress_callback=on_progress,
                visual_out_path=visual_out,
                live_frame_callback=on_live_frame,
            )

            # extract forensic GIF clips while the temp file still exists
            # two-stage approach:
            # 1. try LSTM suspicious events first (most precise timestamps)
            # 2. fall back to YOLO segments when LSTM outputs all-normal
            #    (common when the model is undertrained)
            if result.success and not result.early_exit:
                _SUSP = {"shoplifting", "Looking around", "concealment", "bypass"}

                clip_events = [
                    e for e in (result.behaviour_events or [])
                    if e.behavior_type in _SUSP
                ]

                # fallback: use YOLO segments as clip sources
                # this fires when LSTM gives all-normal but YOLO peak triggered shoplifting
                if not clip_events and result.yolo_segments:
                    from core.result_types import BehaviourEvent
                    clip_events = [
                        BehaviourEvent(
                            behavior_type = s.dominant_class,
                            start_time    = s.start_time,
                            end_time      = s.end_time,
                            confidence    = s.max_shop_conf,
                            probabilities = {},
                        )
                        for s in result.yolo_segments
                        if s.dominant_class in _SUSP or s.max_shop_conf >= 0.30
                    ]

                if clip_events:
                    result.suspicious_frames = self._clip_extractor.extract(
                        tmp_path, clip_events
                    )

            st.session_state.analysis_result = result
        finally:
            # always clear the live preview placeholders after the run
            # and delete the temp file even if the pipeline threw an exception
            _frame_ph.empty()
            _stat_ph.empty()
            _timeline_ph.empty()
            _prog.empty()
            _status.empty()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # analysis output
    def _render_analysis_output(self) -> None:
        result: PipelineResult = st.session_state.analysis_result
        if result is None:
            return
        if not result.success:
            st.error(f"Pipeline error: {result.error}")
            return
        self._analysis_view.render(result)

    # footer
    def _render_footer(self) -> None:
        st.markdown("---")
        st.markdown(
            '<p style="text-align:center;color:#999;font-size:0.85rem;">'
            "Digital Witness - Final Year Project 2026 | "
            "Advisory system only. All alerts require human validation."
            "</p>",
            unsafe_allow_html=True,
        )


# Streamlit runs this module directly (not via __main__) so both paths call run()
if __name__ == "__main__":
    DigitalWitnessApp().run()
else:
    DigitalWitnessApp().run()
