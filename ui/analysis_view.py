"""
ui/analysis_view.py - Renders the video analysis results section.

Shoplifting results split into two tabs:
  - Store Security : Risk Assessment, Detection Statistics, Forensic Evidence
  - Technical Analysis : Step-by-step XAI walkthrough
"""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from core.result_types import PipelineResult, BehaviourEvent


_CLASS_COLOURS = {
    "shoplifting"    : "#e91e63",
    "Looking around" : "#ff9800",
    "Picking-Holding": "#ffc107",
    "normal"         : "#00c853",
    "concealment"    : "#ff5722",
    "bypass"         : "#f44336",
}

_SEV_CSS = {
    "CRITICAL": "alert-critical",
    "HIGH"    : "alert-high",
    "MEDIUM"  : "alert-medium",
    "LOW"     : "alert-low",
    "NONE"    : "alert-none",
}


class AnalysisView:
    """
    Renders the full analysis result section beneath the video uploader.
    Dispatches to _render_early_exit(), _render_normal(), or
    _render_shoplifting() depending on the pipeline outcome.
    """

    def render(self, result: PipelineResult) -> None:
        if not result.success:
            st.error(f"Analysis failed: {result.error or 'Unknown error'}")
            return

        if result.early_exit:
            self._render_early_exit(result)
            return

        if not result.model_trained:
            st.error(
                "**LSTM model not trained.** Results shown are RANDOM and should not be "
                "acted upon. Train the model using the Colab notebook first."
            )
            st.markdown("---")

        st.markdown('<div class="section-header">Analysis Results</div>',
                    unsafe_allow_html=True)

        verdict       = result.lstm_verdict
        peak_conf_val = result.yolo_signal.peak_conf if result.yolo_signal else 0.0

        #  Detection banner 
        if verdict and verdict.is_shoplifting:
            self._render_shoplifting_banner(result, peak_conf_val)
        else:
            self._render_normal_banner(peak_conf_val)

        st.markdown("---")

        #  Normal path: compact summary only ─
        if not (verdict and verdict.is_shoplifting):
            self._render_normal_summary(result)
            return

        #  Shoplifting path: two-tab XAI layout 
        tab_sec, tab_tech = st.tabs(["Store Security", "Technical Analysis"])
        with tab_sec:
            self._render_store_security_tab(result)
        with tab_tech:
            self._render_technical_tab(result)

    #  Banner renderers 

    def _render_shoplifting_banner(self, result: PipelineResult,
                                   peak_conf_val: float) -> None:
        det     = result.detection
        events  = result.behaviour_events
        shop_w  = sum(1 for e in events if e.behavior_type == "shoplifting")
        look_w  = sum(1 for e in events if e.behavior_type == "Looking around")
        persons = det.persons_tracked if det else 0
        products = det.products_detected if det else 0
        st.markdown(f"""
        <div style='background:#cc0000;color:white;padding:1.5rem;
                    border-radius:12px;margin:1rem 0;'>
            <h2 style='margin:0 0 0.5rem 0;'>SHOPLIFTING DETECTED</h2>
            <p style='margin:0 0 0.3rem 0;font-size:1.05rem;'>
                <strong>Verdict:</strong> SHOPLIFTING
                &nbsp;|&nbsp; <strong>Peak Confidence:</strong> {peak_conf_val:.1%}
                &nbsp;|&nbsp; <strong>Threshold:</strong> ≥ 70%
            </p>
            <p style='margin:0;font-size:0.9rem;opacity:0.9;'>
                Shoplifting windows: <strong>{shop_w}</strong>
                &nbsp;·&nbsp; Surveillance windows: <strong>{look_w}</strong>
                &nbsp;·&nbsp; Persons tracked: <strong>{persons}</strong>
                &nbsp;·&nbsp; Product interactions: <strong>{products}</strong>
            </p>
        </div>""", unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        with a1:
            filename = result.video_metadata.filename if result.video_metadata else ""
            st.markdown(f"""
            **Immediate Steps:**
            - Review annotated video below
            - Verify person count ({persons}) matches floor staff observation
            - Cross-check with POS transaction (use POS Audit tab)
            - Do **not** confront - follow store policy
            """)
        with a2:
            alert_id = result.alert.alert_id if result.alert else "No alert generated"
            st.markdown(f"""
            **Evidence to Collect:**
            - Save annotated video clip ({filename})
            - Note timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Export forensic clips from the Store Security tab below
            - Log alert ID: {alert_id}
            """)
        st.warning("**Advisory System** - This is a pattern match, NOT definitive proof. "
                   "Human review is mandatory before any action is taken.")

    def _render_normal_banner(self, peak_conf_val: float) -> None:
        st.markdown(f"""
        <div class='alert-none'>
            <h2 style='margin:0'>NORMAL BEHAVIOR</h2>
            <p style='margin:0.5rem 0 0 0'>
                Peak shoplifting confidence: <strong>{peak_conf_val:.1%}</strong>
                (below 70% threshold)
            </p>
        </div>""", unsafe_allow_html=True)

    #  Normal summary (compact) 

    def _render_normal_summary(self, result: PipelineResult) -> None:
        det = result.detection
        vm  = result.video_metadata
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Persons Tracked",   det.persons_tracked if det else 0)
        c2.metric("Products Detected", det.products_detected if det else 0)
        c3.metric("Frames Processed",  det.frames_processed if det else 0)
        c4.metric("Duration",          f"{vm.duration:.1f}s" if vm else "-")
        st.markdown("---")
        st.markdown("### Video Information")
        c1, c2, c3, c4 = st.columns(4)
        if vm:
            c1.write(f"**File:** {vm.filename}")
            c2.write(f"**Duration:** {vm.duration:.1f}s")
            c3.write(f"**Resolution:** {vm.width}×{vm.height}")
            c4.write(f"**FPS:** {vm.fps:.1f}")
        st.markdown("---")
        st.markdown("""
        <div class='alert-none'>
            <h4 style='margin:0'>No Concerns Detected</h4>
            <p style='margin:0.5rem 0'>Normal shopping behaviour. No immediate concerns.</p>
        </div>""", unsafe_allow_html=True)

    #  Early exit 

    def _render_early_exit(self, result: PipelineResult) -> None:
        st.markdown('<div class="section-header">Analysis Results</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='alert-none'>
            <h2 style='margin:0'>NORMAL - No Suspicious Activity Detected</h2>
        </div>""", unsafe_allow_html=True)
        st.info(result.early_exit_reason or "Video too short for temporal classification.")
        vm = result.video_metadata
        if vm and result.detection:
            c1, c2 = st.columns(2)
            c1.metric("Frames Scanned", result.detection.frames_processed)
            c2.metric("Duration",       f"{vm.duration:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # STORE SECURITY TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _render_store_security_tab(self, result: PipelineResult) -> None:
        intent = result.intent
        det    = result.detection
        sev    = intent.severity if intent else "NONE"
        score  = intent.score if intent else 0.0
        peak_c = result.yolo_signal.peak_conf if result.yolo_signal else 0.0
        div_cls = _SEV_CSS.get(sev, "alert-none")

        # Risk Assessment
        st.markdown("### Risk Assessment")
        st.markdown(f"""
        <div class='{div_cls}'>
            <h3 style='margin:0'>Risk Level: {sev}</h3>
            <p style='margin:0.5rem 0 0 0'>
                Score: {score:.2f} - based on peak detection confidence ({peak_c:.1%}).
            </p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # Detection Statistics
        st.markdown("### Detection Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People Tracked",    det.persons_tracked if det else 0)
        c2.metric("Products Detected", det.products_detected if det else 0)
        c3.metric("Interactions",      det.interactions if det else 0)
        c4.metric("Peak Confidence",   f"{peak_c:.1%}")
        st.markdown("---")

        # Forensic Evidence
        st.markdown("### Forensic Evidence")
        st.caption(
            "Animated clips extracted from the shoplifting window (+ 1 s context). "
            "Each clip is a distinct moment at least 2 s apart - no duplicates."
        )
        clips = result.suspicious_frames
        if clips:
            cols = st.columns(min(len(clips), 4))
            for i, fd in enumerate(clips):
                with cols[i % 4]:
                    if fd.get("gif_bytes"):
                        st.image(fd["gif_bytes"], use_container_width=True)
                    elif fd.get("thumbnail") is not None:
                        st.image(fd["thumbnail"], use_container_width=True)
                    st.caption(
                        f"**{fd['behavior'].upper()}**  \n"
                        f"Frames {fd.get('frame_start', '?')} – {fd.get('frame_end', '?')}  \n"
                        f"{fd.get('clip_start', 0):.1f}s – {fd.get('clip_end', 0):.1f}s  "
                        f"|  {fd['confidence']:.0%} confidence"
                    )
        else:
            st.info("No suspicious clips could be extracted from this video.")
        st.markdown("---")

        # Advisory Summary
        st.markdown("### Advisory Summary")
        alert = result.alert
        if alert:
            _a_cls = "alert-high" if alert.level in {"CRITICAL", "HIGH"} else "alert-medium"
            st.markdown(f"""
            <div class='{_a_cls}'>
                <h4 style='margin:0'>Alert: {alert.alert_id}</h4>
                <p style='margin:0.5rem 0'>{alert.message}</p>
                <p style='margin:0;font-style:italic'>Advisory system - human validation required.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='alert-none'>
                <h4 style='margin:0'>No Alert Generated</h4>
                <p style='margin:0.5rem 0'>No automated alert was raised for this video.</p>
            </div>""", unsafe_allow_html=True)
        st.warning("**Advisory System** - This is a pattern match, NOT definitive proof. "
                   "Human review is mandatory before any action is taken.")

    # ══════════════════════════════════════════════════════════════════════════
    # TECHNICAL ANALYSIS TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _render_technical_tab(self, result: PipelineResult) -> None:
        self._render_step1_yolo(result)
        self._render_step2_mobilenet(result)
        self._render_step3_decision(result)
        self._render_step4_intent(result)
        self._render_step5_timeline(result)

    def _render_step1_yolo(self, result: PipelineResult) -> None:
        sig    = result.yolo_signal
        det    = result.detection
        moments = result.yolo_shop_moments
        st.markdown("### Step 1 - YOLO Detection Evidence")
        st.caption("Primary verdict signal. The peak frame confidence decides the final classification.")

        n_high  = sig.shop_segs_above_50pct if sig else 0
        peak_c  = sig.peak_conf if sig else 0.0
        mean_c  = sig.mean_conf if sig else 0.0
        n_segs  = sig.total_segments if sig else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shoplifting Segments (≥50%)", f"{n_high} / {n_segs}")
        c2.metric("Peak Confidence",  f"{peak_c:.1%}")
        c3.metric("Mean Confidence",  f"{mean_c:.1%}")
        c4.metric("Frames Processed", det.frames_processed if det else 0)

        if moments:
            times = [m["time"] for m in moments]
            confs = [m["conf"] for m in moments]
            fig   = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=confs, mode="markers+lines",
                marker=dict(color=["#cc0000" if c >= 0.70 else "#4caf50" for c in confs],
                            size=8),
                line=dict(color="#cc0000", width=1),
                hovertemplate="t=%{x:.1f}s  conf=%{y:.0%}<extra></extra>",
            ))
            fig.add_hline(y=0.70, line_dash="dash", line_color="#cc0000",
                          annotation_text="70% → SHOPLIFTING")
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=30),
                              xaxis_title="Time (s)",
                              yaxis=dict(tickformat=".0%", range=[0.45, 1.0]),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Red markers ≥ 70% (SHOPLIFTING) | Green markers < 70% (NORMAL) | "
                f"First detection at {moments[0]['time']:.1f}s, "
                f"last at {moments[-1]['time']:.1f}s"
            )
        st.markdown("---")

    def _render_step2_mobilenet(self, result: PipelineResult) -> None:
        st.markdown("### Step 2 - MobileNetV2 Feature Extraction")
        st.caption(
            "1280-dim spatial feature vectors are extracted every 4th frame. "
            "These sequences feed into the BiLSTM temporal classifier."
        )
        vm = result.video_metadata
        if vm:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File",       vm.filename)
            c2.metric("Duration",   f"{vm.duration:.1f}s")
            c3.metric("Resolution", f"{vm.width}×{vm.height}")
            c4.metric("FPS",        f"{vm.fps:.1f}")
        st.markdown("---")

    def _render_step3_decision(self, result: PipelineResult) -> None:
        sig     = result.yolo_signal
        peak_c  = sig.peak_conf if sig else 0.0
        n_high  = sig.shop_segs_above_50pct if sig else 0
        n_segs  = sig.total_segments if sig else 0
        moments = result.yolo_shop_moments
        _n_sec  = len({round(m["time"], 0) for m in moments})
        _spread = "multiple distinct moments" if _n_sec > 1 else "a single moment"

        st.markdown("### Step 3 - Classification Decision")
        st.markdown(f"""
**Decision rule (single peak threshold):**
- Peak shoplifting confidence **≥ 70%** → 🔴 SHOPLIFTING
- Peak shoplifting confidence **< 70%** → 🟢 NORMAL

No segment counting, no vote accumulation - only the highest confidence frame decides.

**Evidence summary:**
- Peak confidence: **{peak_c:.1%}** → verdict: **SHOPLIFTING**
- Frames with shoplifting confidence ≥ 50%: **{len(moments)}** {('spread across ' + _spread) if moments else '(none)'}
- Total 30-frame segments analysed: **{n_segs}** ({n_high} had peak ≥ 50%)
        """)
        st.warning("Advisory system - pattern match only, NOT definitive proof. "
                   "Human review is mandatory.")
        st.markdown("---")

    def _render_step4_intent(self, result: PipelineResult) -> None:
        intent = result.intent
        if not intent:
            return
        st.markdown("### Step 4 - Intent Score")
        st.caption("Composite risk score derived from peak confidence and sustained detection segments.")
        i_c1, i_c2 = st.columns([1, 2])
        with i_c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=intent.score * 100,
                title={"text": "Risk Score"},
                gauge={"axis": {"range": [0, 100]},
                       "bar" : {"color": "#667eea"},
                       "steps": [{"range": [0, 30],  "color": "#d4edda"},
                                  {"range": [30, 50], "color": "#fff3cd"},
                                  {"range": [50, 70], "color": "#ffe4b2"},
                                  {"range": [70, 100],"color": "#ffcccc"}]}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with i_c2:
            st.markdown("**Score Components**")
            comp = intent.components
            if comp:
                fig = go.Figure(go.Bar(
                    x=list(comp.values()), y=list(comp.keys()),
                    orientation="h", marker_color="#667eea",
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20),
                                  xaxis=dict(range=[0, 1], tickformat=".0%"))
                st.plotly_chart(fig, use_container_width=True)
            st.info(intent.explanation)
        st.markdown("---")

    def _render_step5_timeline(self, result: PipelineResult) -> None:
        st.markdown("### Step 5 - Behaviour Timeline")
        st.caption("YOLO 30-frame segments showing detected behaviour classes over time.")

        yolo_segs = result.yolo_segments

        if not yolo_segs:
            st.info("No behaviour segments detected.")
            st.markdown("---")
            return

        fig         = go.Figure()
        seen_labels: set[str] = set()

        for seg in yolo_segs:
            _cls     = seg.dominant_class
            col      = _CLASS_COLOURS.get(_cls, "#667eea")
            show_leg = _cls not in seen_labels
            if show_leg:
                seen_labels.add(_cls)
            fig.add_trace(go.Bar(
                x=[seg.end_time - seg.start_time],
                y=["YOLO Frames"],
                base=[seg.start_time],
                orientation="h",
                marker_color=col,
                name=_cls,
                showlegend=show_leg,
                legendgroup=_cls,
                hovertemplate=(
                    f"<b>{_cls}</b><br>"
                    f"Time: {seg.start_time:.1f}s – {seg.end_time:.1f}s<br>"
                    f"Avg conf: {seg.confidence:.1%}<extra></extra>"
                ),
            ))

        fig.update_layout(
            barmode="overlay",
            title="",
            xaxis=dict(
                title="Time (seconds)",
                range=[0, max((s.end_time for s in yolo_segs), default=60.0) * 1.02],
            ),
            yaxis=dict(showticklabels=False),
            height=160,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        yolo_counts = Counter(s.dominant_class for s in yolo_segs)
        total_segs  = len(yolo_segs)
        st.markdown("**Class distribution across 30-frame segments:**")
        cols = st.columns(len(yolo_counts))
        for ci, (_cls, cnt) in enumerate(sorted(
                yolo_counts.items(), key=lambda x: self._yolo_priority(x[0]))):
            cols[ci].metric(_cls, f"{cnt} seg", f"{cnt / total_segs:.0%}")

        with st.expander("View YOLO Segment Detail"):
            yrows = [{"Class"   : s.dominant_class,
                      "Start"   : f"{s.start_time:.1f}s",
                      "End"     : f"{s.end_time:.1f}s",
                      "Avg Conf": f"{s.confidence:.1%}",
                      "Peak Shop Conf": f"{s.max_shop_conf:.1%}"}
                     for s in yolo_segs]
            st.dataframe(pd.DataFrame(yrows), use_container_width=True, hide_index=True)
        st.markdown("---")

    #  Static helpers 

    @staticmethod
    def _yolo_priority(cls: str) -> int:
        """Sort order for YOLO class display (most suspicious first)."""
        return {"shoplifting": 0, "Looking around": 1,
                "Picking-Holding": 2, "normal": 3}.get(cls, 4)
