"""
ui/analysis_view.py - Renders the video analysis results section.

Shoplifting results split into two tabs:
  - Store Security : Risk Assessment, Detection Statistics, Forensic Evidence
  - Technical Analysis : Step-by-step XAI walkthrough (steps 1-5)

The XAI (Explainable AI) presentation is inspired by the LIME and SHAP
literature which argues that end users need to understand why a model made
a decision, not just what it decided:
    Ribeiro, M.T. et al. (2016). "Why Should I Trust You?": Explaining the
    Predictions of Any Classifier. ACM SIGKDD.
    https://arxiv.org/abs/1602.04938

Plotly is used for all charts in this file:
    Plotly Technologies Inc. (2015). Collaborative data science.
    https://plotly.com
    MIT License.

NumPy and Pandas for data wrangling:
    Harris et al. (2020). Array programming with NumPy. Nature 585, 357-362.
    McKinney, W. (2010). Data Structures for Statistical Computing in Python. SciPy.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from core.result_types import PipelineResult, BehaviourEvent


# colour map for YOLO behaviour classes - matches the bounding box colours in video_processor.py
# shoplifting is pink/red, looking around is orange, normal is green
_CLASS_COLOURS = {
    "shoplifting"    : "#e91e63",
    "Looking around" : "#ff9800",
    "Picking-Holding": "#ffc107",
    "normal"         : "#00c853",
    "concealment"    : "#ff5722",
    "bypass"         : "#f44336",
}

# maps severity label to the CSS class defined in styles.py
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
    Dispatches to _render_early_exit(), _render_normal_summary(), or
    the full shoplifting two-tab layout depending on the pipeline outcome.
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

        # top banner - red for shoplifting, green for normal
        if verdict and verdict.is_shoplifting:
            self._render_shoplifting_banner(result, peak_conf_val)
        else:
            self._render_normal_banner(peak_conf_val)

        st.markdown("---")

        # normal path: compact summary + XAI explanation, then return
        if not (verdict and verdict.is_shoplifting):
            self._render_normal_summary(result)
            self._render_xai_summary(result)
            return

        # shoplifting path: two-tab layout for deeper detail
        tab_sec, tab_tech = st.tabs(["Store Security", "Technical Analysis"])
        with tab_sec:
            self._render_store_security_tab(result)
        with tab_tech:
            self._render_technical_tab(result)

    # banner renderers

    def _render_shoplifting_banner(self, result: PipelineResult,
                                   peak_conf_val: float) -> None:
        det      = result.detection
        events   = result.behaviour_events
        shop_w   = sum(1 for e in events if e.behavior_type == "shoplifting")
        look_w   = sum(1 for e in events if e.behavior_type == "Looking around")
        persons  = det.persons_tracked if det else 0
        products = det.products_detected if det else 0
        st.markdown(f"""
        <div style='background:#cc0000;color:white;padding:1.5rem;
                    border-radius:12px;margin:1rem 0;'>
            <h2 style='margin:0 0 0.5rem 0;'>SHOPLIFTING DETECTED</h2>
            <p style='margin:0 0 0.3rem 0;font-size:1.05rem;'>
                <strong>Verdict:</strong> SHOPLIFTING
                &nbsp;|&nbsp; <strong>Peak Confidence:</strong> {peak_conf_val:.1%}
                &nbsp;|&nbsp; <strong>Threshold:</strong> 70%
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

    # =========================================================================
    # XAI SUMMARY — why did the model say SHOPLIFTING?
    # =========================================================================

    def _render_xai_summary(self, result: PipelineResult) -> None:
        """
        Plain-English explanation of the decision, shown immediately after the
        verdict banner so the user never has to hunt for the reason.

        Three-column layout:
          col 1 — WHAT triggered it   (YOLO detection evidence)
          col 2 — WHEN it happened    (BiLSTM attention peak moments)
          col 3 — HOW serious it is   (Intent score component breakdown)
        """
        is_shop = bool(result.lstm_verdict and result.lstm_verdict.is_shoplifting)

        if is_shop:
            st.markdown("### Why did the model flag this video?")
        else:
            st.markdown("### Why did the model classify this as Normal?")

        sig     = result.yolo_signal
        intent  = result.intent
        attn    = result.lstm_attention_timeline
        moments = result.yolo_shop_moments

        col1, col2, col3 = st.columns(3)

        # ── WHAT: YOLO confidence vs threshold ───────────────────────────────
        with col1:
            peak_c = sig.peak_conf if sig else 0.0
            st.markdown("#### YOLO Confidence")
            if is_shop:
                first_t = moments[0]["time"] if moments else None
                peak_t  = max(moments, key=lambda m: m["conf"])["time"] if moments else None
                st.markdown(f"""
**Peak confidence: `{peak_c:.1%}`** — above the 70% threshold.

YOLO crossed the threshold at `{peak_t:.1f}s`, triggering the SHOPLIFTING verdict.

First detection: `{first_t:.1f}s`
Peak detection: `{peak_t:.1f}s`
Frames flagged: `{sig.frames_with_shoplifting if sig else 0}`
""" if first_t is not None else f"**Peak confidence: `{peak_c:.1%}`** (above 70% threshold).")
            else:
                st.markdown(f"""
**Peak confidence: `{peak_c:.1%}`** — below the 70% threshold.

YOLO never reached the 70% decision threshold in any frame of this video,
so no shoplifting verdict was triggered.

Frames with any shoplifting detection: `{sig.frames_with_shoplifting if sig else 0}`
""")

        # ── WHEN: BiLSTM attention peaks ──────────────────────────────────────
        with col2:
            st.markdown("#### BiLSTM Focus Moments")
            if attn:
                weights = np.array([p["weight"] for p in attn])
                times   = [p["time"] for p in attn]
                thresh  = weights.mean() + weights.std()
                top_moments = sorted(
                    [(times[i], weights[i]) for i in range(len(times))
                     if weights[i] >= thresh],
                    key=lambda x: x[1], reverse=True
                )[:5]

                label_text = ("Moments the model weighted most — these frames drove "
                              "the shoplifting classification:" if is_shop else
                              "Moments the model paid most attention to — no suspicious "
                              "pattern was found in these frames:")
                st.markdown(label_text)
                bar_colour = "#cc0000" if is_shop else "#43a047"
                for t, w in top_moments:
                    bar = "█" * int(w / weights.max() * 10)
                    st.markdown(f"- **{t:.1f}s** &nbsp; `{bar}` &nbsp; weight={w:.4f}")

                fig = go.Figure(go.Bar(
                    x=times, y=weights.tolist(),
                    marker_color=[bar_colour if w >= thresh else "#b0bec5"
                                  for w in weights],
                ))
                fig.update_layout(
                    height=130,
                    margin=dict(l=0, r=0, t=4, b=24),
                    xaxis_title="Time (s)",
                    yaxis_title="",
                    showlegend=False,
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Attention timeline not available.")

        # ── HOW SERIOUS: intent components ───────────────────────────────────
        with col3:
            st.markdown("#### Risk Assessment")
            if intent:
                sev_colour = {
                    "CRITICAL": "#c62828", "HIGH": "#e53935",
                    "MEDIUM":   "#fb8c00", "LOW":  "#fdd835", "NONE": "#43a047",
                }.get(intent.severity, "#78909c")

                st.markdown(
                    f"**Overall risk: "
                    f"<span style='color:{sev_colour}'>{intent.severity}</span> "
                    f"({intent.score:.2f} / 1.00)**",
                    unsafe_allow_html=True,
                )
                comp = intent.components
                if comp:
                    # map component names to human-readable labels
                    labels = {
                        "concealment": "Concealment behaviour",
                        "bypass":      "Exit/bypass behaviour",
                        "duration":    "Sustained detection",
                    }
                    for key, score in sorted(comp.items(),
                                             key=lambda x: x[1], reverse=True):
                        label  = labels.get(key, key)
                        filled = int(score * 10)
                        bar    = "█" * filled + "░" * (10 - filled)
                        st.markdown(f"- **{label}**: `{bar}` `{score:.2f}`")

                st.caption(intent.explanation)
            else:
                st.info("Intent score not available.")

    # normal summary (compact, no XAI tabs needed)

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
            c3.write(f"**Resolution:** {vm.width}x{vm.height}")
            c4.write(f"**FPS:** {vm.fps:.1f}")
        st.markdown("---")
        st.markdown("""
        <div class='alert-none'>
            <h4 style='margin:0'>No Concerns Detected</h4>
            <p style='margin:0.5rem 0'>Normal shopping behaviour. No immediate concerns.</p>
        </div>""", unsafe_allow_html=True)

    # early exit (video too short for temporal classification)

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

    # =========================================================================
    # STORE SECURITY TAB
    # =========================================================================

    def _render_store_security_tab(self, result: PipelineResult) -> None:
        intent  = result.intent
        det     = result.detection
        sev     = intent.severity if intent else "NONE"
        score   = intent.score if intent else 0.0
        peak_c  = result.yolo_signal.peak_conf if result.yolo_signal else 0.0
        div_cls = _SEV_CSS.get(sev, "alert-none")

        # risk assessment
        st.markdown("### Risk Assessment")
        st.markdown(f"""
        <div class='{div_cls}'>
            <h3 style='margin:0'>Risk Level: {sev}</h3>
            <p style='margin:0.5rem 0 0 0'>
                Score: {score:.2f} - based on peak detection confidence ({peak_c:.1%}).
            </p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # detection statistics
        st.markdown("### Detection Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People Tracked",    det.persons_tracked if det else 0)
        c2.metric("Products Detected", det.products_detected if det else 0)
        c3.metric("Interactions",      det.interactions if det else 0)
        c4.metric("Peak Confidence",   f"{peak_c:.1%}")
        st.markdown("---")

        # forensic evidence - GIF clips from the shoplifting windows
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
                        f"Frames {fd.get('frame_start', '?')} - {fd.get('frame_end', '?')}  \n"
                        f"{fd.get('clip_start', 0):.1f}s - {fd.get('clip_end', 0):.1f}s  "
                        f"|  {fd['confidence']:.0%} confidence"
                    )
        else:
            st.info("No suspicious clips could be extracted from this video.")
        st.markdown("---")

        # advisory summary with alert details
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

    # =========================================================================
    # TECHNICAL ANALYSIS TAB
    # =========================================================================

    def _render_technical_tab(self, result: PipelineResult) -> None:
        # Order: XAI summary first (most relevant), then supporting detail
        self._render_xai_summary(result)        # Why the model decided
        self._render_step2_mobilenet(result)    # BiLSTM attention chart (detailed)
        self._render_step4_intent(result)       # Intent score breakdown
        self._render_step1_yolo(result)         # YOLO raw detection evidence
        self._render_step3_decision(result)     # Decision rule explanation
        self._render_step5_timeline(result)     # Full behaviour timeline

    def _render_step1_yolo(self, result: PipelineResult) -> None:
        sig     = result.yolo_signal
        det     = result.detection
        moments = result.yolo_shop_moments
        st.markdown("### YOLO Detection Evidence")
        st.caption("Primary verdict signal. The peak frame confidence decides the final classification.")

        n_high = sig.shop_segs_above_50pct if sig else 0
        peak_c = sig.peak_conf if sig else 0.0
        mean_c = sig.mean_conf if sig else 0.0
        n_segs = sig.total_segments if sig else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shoplifting Segments (>=50%)", f"{n_high} / {n_segs}")
        c2.metric("Peak Confidence",  f"{peak_c:.1%}")
        c3.metric("Mean Confidence",  f"{mean_c:.1%}")
        c4.metric("Frames Processed", det.frames_processed if det else 0)

        if moments:
            times = [m["time"] for m in moments]
            confs = [m["conf"] for m in moments]
            # Plotly scatter - Plotly Technologies Inc. (2015)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=confs, mode="markers+lines",
                marker=dict(color=["#cc0000" if c >= 0.70 else "#4caf50" for c in confs],
                            size=8),
                line=dict(color="#cc0000", width=1),
                hovertemplate="t=%{x:.1f}s  conf=%{y:.0%}<extra></extra>",
            ))
            # horizontal line at 70% threshold - above this triggers SHOPLIFTING verdict
            fig.add_hline(y=0.70, line_dash="dash", line_color="#cc0000",
                          annotation_text="70% -> SHOPLIFTING")
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=30),
                              xaxis_title="Time (s)",
                              yaxis=dict(tickformat=".0%", range=[0.45, 1.0]),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Red markers >= 70% (SHOPLIFTING) | Green markers < 70% (NORMAL) | "
                f"First detection at {moments[0]['time']:.1f}s, "
                f"last at {moments[-1]['time']:.1f}s"
            )
        st.markdown("---")

    def _render_step2_mobilenet(self, result: PipelineResult) -> None:
        st.markdown("### BiLSTM Temporal Attention (XAI)")
        st.caption(
            "Each bar shows how much the BiLSTM weighted that moment when making its "
            "decision. Taller bars = frames the model considered most suspicious. "
            "Weights are softmax-normalised and averaged across overlapping windows."
        )

        timeline = result.lstm_attention_timeline
        if not timeline:
            st.info("Attention data not available for this video.")
            st.markdown("---")
            return

        times   = [p["time"]   for p in timeline]
        weights = [p["weight"] for p in timeline]

        # threshold: frames above mean + 1 std are highlighted as key moments
        arr      = np.array(weights)
        mean_w   = float(arr.mean())
        thresh_w = float(arr.mean() + arr.std())
        colours  = ["#cc0000" if w >= thresh_w else
                    "#ff9800" if w >= mean_w   else
                    "#b0bec5"
                    for w in weights]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=times,
            y=weights,
            marker_color=colours,
            hovertemplate="t=%{x:.1f}s  attention=%{y:.4f}<extra></extra>",
        ))
        # mark the decision threshold line
        fig.add_hline(
            y=thresh_w, line_dash="dot", line_color="#cc0000",
            annotation_text="high-attention threshold",
            annotation_position="top right",
        )
        fig.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis_title="Time in video (seconds)",
            yaxis_title="Attention weight",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # pick top-3 moments for a plain-English summary
        top_idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:3]
        top_times = sorted([times[i] for i in top_idx])
        st.caption(
            f"**Key moments flagged by the model:** "
            + "  ·  ".join(f"{t:.1f}s" for t in top_times)
            + "  (red bars above dashed line)"
        )

        with st.expander("What does this chart mean?"):
            st.markdown("""
**Temporal Attention — the XAI explanation for the BiLSTM decision.**

The BiLSTM+Attention classifier does not treat every frame equally.
Inside the model, a learned attention layer assigns a weight to each of the
45 frames in every sliding window. These weights sum to 1.0 (softmax) and
represent how much the model "focused" on each moment before producing its
classification.

**How to read this chart:**
- Each bar = one extracted feature frame (every 4th video frame).
- Height = average attention weight across all overlapping windows that covered that frame.
- **Red bars** exceed mean + 1σ — these are the moments the model relied on most.
- **Orange bars** are above-average attention — secondary evidence.
- **Grey bars** contributed little to the final decision.

**Why attention weights are valid XAI:**
Attention-based explainability is an intrinsic, model-internal technique —
unlike post-hoc approximations (LIME, SHAP), the weights are a direct product
of the forward pass. They do not approximate the model; they *are* the model's
internal reasoning about temporal importance.

Reference: Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation
by Jointly Learning to Align and Translate. ICLR 2015.
https://arxiv.org/abs/1409.0473
            """)
        st.markdown("---")

    def _render_step3_decision(self, result: PipelineResult) -> None:
        sig     = result.yolo_signal
        peak_c  = sig.peak_conf if sig else 0.0
        n_high  = sig.shop_segs_above_50pct if sig else 0
        n_segs  = sig.total_segments if sig else 0
        moments = result.yolo_shop_moments
        # count how many distinct seconds had a shoplifting detection
        _n_sec  = len({round(m["time"], 0) for m in moments})
        _spread = "multiple distinct moments" if _n_sec > 1 else "a single moment"

        st.markdown("### Classification Decision")
        st.markdown(f"""
**Decision rule (single peak threshold):**
- Peak shoplifting confidence **>= 70%** -> SHOPLIFTING
- Peak shoplifting confidence **< 70%** -> NORMAL

No segment counting, no vote accumulation - only the highest confidence frame decides.

**Evidence summary:**
- Peak confidence: **{peak_c:.1%}** -> verdict: **SHOPLIFTING**
- Frames with shoplifting confidence >= 50%: **{len(moments)}** {('spread across ' + _spread) if moments else '(none)'}
- Total 30-frame segments analysed: **{n_segs}** ({n_high} had peak >= 50%)
        """)
        st.warning("Advisory system - pattern match only, NOT definitive proof. "
                   "Human review is mandatory.")
        st.markdown("---")

    def _render_step4_intent(self, result: PipelineResult) -> None:
        intent = result.intent
        if not intent:
            return
        st.markdown("### Intent Score")
        st.caption("Composite risk score derived from peak confidence and sustained detection segments.")
        i_c1, i_c2 = st.columns([1, 2])
        with i_c1:
            # gauge chart using Plotly indicator
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
        st.markdown("### Behaviour Timeline")
        st.caption("YOLO 30-frame segments showing detected behaviour classes over time.")

        yolo_segs = result.yolo_segments

        if not yolo_segs:
            st.info("No behaviour segments detected.")
            st.markdown("---")
            return

        fig         = go.Figure()
        seen_labels: set[str] = set()

        # build the gantt-style horizontal bar chart from YOLO segments
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
                    f"Time: {seg.start_time:.1f}s - {seg.end_time:.1f}s<br>"
                    f"Avg conf: {seg.confidence:.1%}<extra></extra>"
                ),
            ))

        # x-axis starts from 0 so there's a visible gap before the first detection
        max_t = max((s.end_time for s in yolo_segs), default=60.0)
        fig.update_layout(
            barmode="overlay",
            title="",
            xaxis=dict(
                title="Time (seconds)",
                range=[0, max_t * 1.02],
            ),
            yaxis=dict(showticklabels=False),
            height=160,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        # add some top padding between the heading and the chart
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        # class distribution summary below the chart
        yolo_counts = Counter(s.dominant_class for s in yolo_segs)
        total_segs  = len(yolo_segs)
        st.markdown("**Class distribution across 30-frame segments:**")
        cols = st.columns(len(yolo_counts))
        for ci, (_cls, cnt) in enumerate(sorted(
                yolo_counts.items(), key=lambda x: self._yolo_priority(x[0]))):
            cols[ci].metric(_cls, f"{cnt} seg", f"{cnt / total_segs:.0%}")

        with st.expander("View YOLO Segment Detail"):
            yrows = [{"Class"          : s.dominant_class,
                      "Start"          : f"{s.start_time:.1f}s",
                      "End"            : f"{s.end_time:.1f}s",
                      "Avg Conf"       : f"{s.confidence:.1%}",
                      "Peak Shop Conf" : f"{s.max_shop_conf:.1%}"}
                     for s in yolo_segs]
            st.dataframe(pd.DataFrame(yrows), use_container_width=True, hide_index=True)
        st.markdown("---")

    # static helpers

    @staticmethod
    def _yolo_priority(cls: str) -> int:
        """Sort order for YOLO class display - most suspicious shown first."""
        return {"shoplifting": 0, "Looking around": 1,
                "Picking-Holding": 2, "normal": 3}.get(cls, 4)
