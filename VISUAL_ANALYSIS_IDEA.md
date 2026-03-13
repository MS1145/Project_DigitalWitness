# Visual Real-Time Analysis Display — Implementation Idea

## Concept

While the pipeline is processing a video, instead of showing a plain spinner,
display a **live preview window** that shows the video playing with overlays
from each model stage as the frames are processed. This gives the examiner
a visual understanding of what the system is doing and makes the wait feel
purposeful.

---

## Proposed UI Layout (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│  [Video Preview — frame N of M]                             │
│  ┌───────────────────────────────┐  ┌─────────────────────┐ │
│  │                               │  │  LIVE STATS         │ │
│  │   [annotated video frame]     │  │  ─────────────────  │ │
│  │   YOLO bounding boxes         │  │  Persons tracked: 2 │ │
│  │   MobileNetV2 label overlay   │  │  Suspicious acts: 3 │ │
│  │   Confidence score            │  │  POS matches: 1     │ │
│  │                               │  │  Intent score: 0.72 │ │
│  └───────────────────────────────┘  └─────────────────────┘ │
│                                                              │
│  [Timeline bar — colour-coded per frame]                     │
│  ████████████████░░░░░░░░░░░░░░░░░░░░░░  38% processed      │
│  GREEN = Normal   RED = Suspicious   GRAY = No detection     │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Overlay Description

### Stage 1 — YOLO Person Detection
- Draw **green bounding boxes** around each detected person.
- Draw **blue bounding boxes** around detected products (basket, handbag, etc.).
- Show **track ID** (e.g., `Person #1`, `Person #2`) above each box.
- Show YOLO confidence score in small text.

### Stage 2 — MobileNetV2 Frame Classification
- Overlay a **label badge** in the top-left of the frame:
  - `NORMAL` → green badge
  - `SHOPLIFTING` → red badge
- Show the softmax confidence below the badge (e.g., `94.2% shoplifting`).

### Stage 3 — BiLSTM Sequence Decision
- Once enough frames are buffered (sliding window), show the **sequence verdict**
  in the top-right corner:
  - `SEQ: SUSPICIOUS` → red border around the frame
  - `SEQ: NORMAL` → neutral border
- Updates every N frames (window slide step).

### Stage 4 — POS Verification (Visual Idea)
POS data is text-based so the visual approach needs some creativity:

**Option A — Receipt timeline:**
- Show a vertical scrolling receipt on the right panel as items from POS are
  listed. Highlight the timestamp that matches the current video frame time.
  Unmatched time windows appear in red.

**Option B — Match indicator:**
- Show a simple icon: `✓ POS match` (green) or `✗ No POS record` (red) for the
  current time window.

**Option C — Heatmap strip:**
- Below the video, show a horizontal heatmap strip where each second of video
  is coloured by whether a POS transaction exists in that window.

---

## Implementation Approach

### Backend: `run_pipeline_streaming()`

Modify `run_pipeline()` to **yield** intermediate results instead of returning
everything at the end:

```python
def run_pipeline_streaming(video_path, pos_path=None):
    """
    Generator version of run_pipeline.
    Yields dicts with frame data as processing happens.
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        yolo_result = yolo_model.track(frame, persist=True, conf=YOLO_CONF)[0]
        annotated = draw_yolo_boxes(frame, yolo_result)

        # MobileNetV2 frame classification (if person detected)
        mobilenet_label, mobilenet_conf = None, None
        if has_person(yolo_result):
            crop = crop_person(frame, yolo_result)
            mobilenet_label, mobilenet_conf = classify_frame(crop)
            annotated = draw_mobilenet_overlay(annotated, mobilenet_label, mobilenet_conf)

        # BiLSTM — only update every WINDOW_STEP frames
        bilstm_verdict = None
        if len(feature_buffer) >= SEQUENCE_LEN:
            bilstm_verdict = run_bilstm_window(feature_buffer)
            annotated = draw_bilstm_overlay(annotated, bilstm_verdict)

        # Encode frame to JPEG for streaming
        _, buf = cv2.imencode('.jpg', annotated)
        frame_b64 = base64.b64encode(buf).decode()

        yield {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "frame_b64": frame_b64,
            "mobilenet_label": mobilenet_label,
            "mobilenet_conf": mobilenet_conf,
            "bilstm_verdict": bilstm_verdict,
            "persons_tracked": count_persons(yolo_result),
            "suspicious_count": suspicious_count,
            "intent_score": running_intent_score,
        }

        frame_idx += 1

    cap.release()
    # Final result yielded last
    yield {"final": True, "result": build_final_result(...)}
```

### Frontend: Streamlit Rendering

```python
# In the Streamlit UI (video analysis tab):

if st.button("Analyse Video"):
    placeholder = st.empty()
    timeline_placeholder = st.empty()
    stats_placeholder = st.empty()

    frame_labels = []

    for update in run_pipeline_streaming(video_path, pos_path):
        if update.get("final"):
            st.session_state["last_result"] = update["result"]
            break

        # Display annotated frame
        with placeholder.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(
                    base64.b64decode(update["frame_b64"]),
                    caption=f"Frame {update['frame_idx']} / {update['total_frames']}",
                    use_column_width=True,
                )
            with col2:
                st.metric("Persons Tracked", update["persons_tracked"])
                st.metric("Suspicious Acts", update["suspicious_count"])
                if update["mobilenet_conf"]:
                    label = update["mobilenet_label"]
                    conf = update["mobilenet_conf"]
                    colour = "red" if label == "shoplifting" else "green"
                    st.markdown(
                        f"<span style='color:{colour}; font-size:18px;'>"
                        f"**{label.upper()}** ({conf:.1%})</span>",
                        unsafe_allow_html=True,
                    )

        # Update colour-coded timeline strip
        frame_labels.append(update.get("mobilenet_label", "unknown"))
        with timeline_placeholder:
            render_timeline_strip(frame_labels, update["total_frames"])

    # Show final results after processing
    render_final_result(st.session_state["last_result"])
```

### Timeline Strip Helper

```python
def render_timeline_strip(frame_labels, total_frames):
    """Render a horizontal colour-coded bar showing per-frame classification."""
    colours = {
        "normal": "#28a745",
        "shoplifting": "#dc3545",
        "unknown": "#6c757d",
    }
    # Build HTML bar
    bar_html = '<div style="display:flex; width:100%; height:16px;">'
    for label in frame_labels:
        pct = 100 / max(total_frames, 1)
        colour = colours.get(label, "#6c757d")
        bar_html += f'<div style="flex:{pct}; background:{colour};"></div>'
    bar_html += '</div>'
    # Gray placeholder for unprocessed frames
    remaining = total_frames - len(frame_labels)
    if remaining > 0:
        pct = (remaining / total_frames) * 100
        bar_html += f'<div style="width:{pct}%; height:16px; background:#e9ecef;"></div>'
    st.markdown(bar_html, unsafe_allow_html=True)
    progress = len(frame_labels) / max(total_frames, 1)
    st.progress(progress, text=f"{progress:.0%} processed")
```

---

## POS Verification Visual — Recommended Approach

Since POS data is a CSV/JSON of `(timestamp, item, price)`, map it to the video
timeline:

```
Video timeline  |----[0s]----------[10s]----------[20s]----------[30s]----|
POS events      |              [item scanned]            [payment]         |
Suspicious zone |         [███ shoplifting window ███]                     |
```

- Render this as an SVG or HTML timeline in the sidebar.
- Highlight overlapping windows where **shoplifting was detected** but
  **no POS transaction exists**.
- This is the core "proof" the examiner needs — visual evidence of concealment.

---

## Technical Challenges to Solve

| Challenge | Solution |
|---|---|
| Streamlit doesn't natively stream frames | Use `st.empty()` + generator loop |
| Processing is slow — UI freezes | Run pipeline in a background thread, push frames to a queue read by UI |
| Frame-by-frame JPEG encoding is slow | Downsample display frames (e.g., show every 3rd frame in preview) |
| BiLSTM needs full sequence before deciding | Buffer features, update verdict every N frames |
| POS timestamps may not align with video frame timestamps | Map by elapsed time offset, not absolute time |

---

## Alternative: Pre-render the annotated video, then play it

Instead of live streaming, run the full pipeline first (with progress bar),
save the annotated video to a temp file, then play it in the UI:

```python
# After processing:
out_path = "temp_annotated.mp4"
render_annotated_video(video_path, frame_results, out_path)
st.video(out_path)  # Streamlit native video player
```

**Pros:** Simpler to implement, no threading required.
**Cons:** User sees nothing until 100% done; no live feel.

This could be a good **Phase 1** step before building the full streaming UI.

---

## Recommended Implementation Order

1. **Phase 1** — Pre-render annotated video
   Run pipeline → save annotated MP4 → play with `st.video()`.
   Shows YOLO boxes + MobileNetV2 labels baked into frames.

2. **Phase 2** — Add live timeline strip
   During processing, update a progress bar and colour-coded frame strip
   so the examiner sees activity per second.

3. **Phase 3** — Full streaming preview
   Refactor `run_pipeline()` to a generator, stream frames to `st.image()`
   inside `st.empty()` for real-time feel.

4. **Phase 4** — POS overlay
   Add SVG/HTML timeline showing POS events vs suspicious windows.
