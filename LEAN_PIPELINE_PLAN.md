# Lean Pipeline B — Implementation Plan
## YOLO Detections → BiLSTM Direct (No MobileNetV2)

**Goal:** Remove MobileNetV2 entirely. Feed YOLO's per-frame detection outputs directly
into a smaller BiLSTM. Same accuracy target, lower memory, faster inference, suitable
for low-spec hardware (Raspberry Pi 4 / basic CCTV server).

---

## Why This Is Better for Low-Spec Hardware

| | Current Pipeline | Lean Pipeline B |
|---|---|---|
| Models loaded | YOLO + MobileNetV2 + BiLSTM | YOLO + BiLSTM only |
| RAM at inference | ~800 MB | ~150 MB |
| Feature vector | 1280-dim | 16-dim |
| BiLSTM params | ~4M | ~200K |
| MobileNetV2 load time | ~2s | — |
| Inference per frame | YOLO + CNN crop | YOLO only |

---

## Phase 1 — New Training Notebook
**File:** `DigitalWitness_LeanB_Pipeline.ipynb`

### Cell 1 — Imports + Config
- Reuse same ROOT / MODELS_DIR / TRAIN_VIDEOS paths
- New constants:
  - `LEAN_SEQ_DIR = data/lean_sequences/`
  - `LEAN_SAVE    = models/bilstm_lean_dw.pt`
  - `LEAN_INFO    = models/bilstm_lean_dw_info.json`
  - `LEAN_FEAT_DIM = 16`
  - `LSTM_SEQ_LEN  = 45` (same 7.5s window)
  - `LEAN_HIDDEN   = 128` (half of current 256)
  - `SMOKE_TEST    = False`

### Cell 2 — Load YOLO + Define Feature Extractor
Load existing `yolo26_dw_v2.pt`. Define `extract_lean_features(results, w, h)` that
converts YOLO detection output for one frame into a **16-dim float32 vector**:

```
feat[ 0] = max shoplifting confidence
feat[ 1] = max looking-around confidence
feat[ 2] = max picking-holding confidence
feat[ 3] = max normal confidence
feat[ 4] = shoplifting detection count / 5  (clamped 0-1)
feat[ 5] = looking-around count / 5
feat[ 6] = picking-holding count / 5
feat[ 7] = total detections / 10
feat[ 8] = centre-x of highest-conf detection (normalised 0-1, default 0.5)
feat[ 9] = centre-y of highest-conf detection (normalised 0-1, default 0.5)
feat[10] = bbox width of highest-conf detection (normalised, 0 if none)
feat[11] = bbox height (normalised, 0 if none)
feat[12] = 1.0 if any detection present
feat[13] = 1.0 if shoplifting OR looking-around detected
feat[14] = 1.0 if shoplifting confidence > 0.5
feat[15] = (n_shoplifting + n_looking) / total  — suspicious ratio
```
If no detections: all zeros except feat[8]=feat[9]=0.5.

### Cell 3 — Extract YOLO Feature Sequences from Videos
- Loop over `TRAIN_VIDEOS/normal/` and `TRAIN_VIDEOS/shoplifting/`
- For each video: read frames at 6fps (step = src_fps / 6), run YOLO predict,
  call `extract_lean_features`, build (T, 16) array
- Save as `lean_sequences/cls/video_stem.npy`
- Skip if file already exists (re-run safe)
- Print per-class sequence count

### Cell 4 — Dataset + 80/10/10 Split
- `LeanSequenceDataset`: sliding windows (SEQ_LEN=45, STRIDE=15), zero-pad short
- Stratified 80/10/10 split on sequence FILES (not windows) — prevents data leakage
- Save test split to `models/bilstm_lean_test_split.json`
- WeightedRandomSampler for shoplifting minority class
- CrossEntropyLoss with class weights
- Print class distribution per split

### Cell 5 — Lean BiLSTM Model
Same architecture as current BiLSTM but:
- `input_size = 16` (not 1280)
- `hidden_size = 128` (not 256)
- Same temporal attention mechanism
- Same classifier: Linear(256, 2)
- Print parameter count (expected ~200K vs ~4M)

### Cell 6 — Training Loop
- AdamW optimiser, StepLR (halve every 8 epochs, same as current)
- Early stopping patience = 7
- Save best checkpoint as:
  ```python
  {
    'state_dict': model.state_dict(),
    'feat_dim': 16,
    'hidden': 128,
    'seq_len': 45,
    'pipeline': 'lean_b'
  }
  ```
- SMOKE_TEST = 1-epoch quick check

### Cell 7 — Evaluation
- Load test split, run inference on all windows
- Print classification report (precision / recall / F1)
- Confusion matrix → save `outputs/cases/lean_b_confusion.png`
- Training curves → save `outputs/cases/lean_b_curves.png`

### Cell 8 — Save Model Info JSON
```json
{
  "pipeline": "lean_b",
  "feat_dim": 16,
  "hidden": 128,
  "seq_len": 45,
  "stride": 15,
  "best_val_acc": <float>,
  "test_acc": <float>,
  "classes": ["normal", "shoplifting"],
  "notes": "YOLO->BiLSTM direct. No MobileNetV2. Low-spec device optimised."
}
```

---

## Phase 2 — app.py Integration
**No UI changes. No POS changes. Only model loading + feature extraction.**

### 2a — New Constants (after existing LSTM_PATH line)
```python
BILSTM_LEAN_PATH = MODELS_DIR / "bilstm_lean_dw.pt"
LEAN_FEAT_DIM    = 16
```

### 2b — `_build_bilstm()` — Add `input_size` Parameter
Change signature from fixed `input_size=1280` inside nn.LSTM to a parameter:
```python
def _build_bilstm(num_classes=2, hidden=256, layers=2, dropout=0.3, input_size=1280):
```

### 2c — `run_pipeline()` — Auto-detect Lean Model
Priority logic:
```
if bilstm_lean_dw.pt exists → USE_LEAN = True  (skip MobileNetV2 entirely)
else                         → USE_LEAN = False (existing pipeline unchanged)
```
Load lean model with correct hidden=128, input_size=16 from checkpoint metadata.

### 2d — `run_pipeline()` — Skip MobileNetV2 Loading
Wrap MobileNetV2 + xform setup in `if not USE_LEAN`.

### 2e — `run_pipeline()` — Feature Extraction in Processing Loop
Add `_extract_lean_feat()` helper inside `run_pipeline()` (local function).
Replace per-frame feature extraction block:
```
if USE_LEAN:
    append _extract_lean_feat(yolo_results) every yolo_step frame
else:
    append mnet_model.extract_features() every feat_step frame  ← unchanged
```
Rolling live BiLSTM badge: unchanged (both branches feed all_feats).

### 2f — Fix feats_arr Dimension
```python
_fdim     = LEAN_FEAT_DIM if USE_LEAN else MOBILENET_DIM
feats_arr = np.array(all_feats) if all_feats else np.zeros((1, _fdim))
```
Padding zero arrays also use `_fdim`.

### 2g — Fix Window Timing
```python
step_size = yolo_step if USE_LEAN else feat_step
"start": start * step_size / fps_val
"end":   (start + LSTM_SEQ_LEN) * step_size / fps_val
```

### 2h — `check_model_exists()`
```python
return LSTM_PATH.exists() or BILSTM_LEAN_PATH.exists()
```

---

## POS Integration — No Changes Needed
POS audit uses `product_pickup_events` which is populated from YOLO detections
(Picking-Holding class boxes). Since YOLO still runs in the lean pipeline,
POS cross-referencing works identically.

---

## Execution Order

- [ ] **Step 1** — Create `DigitalWitness_LeanB_Pipeline.ipynb` (notebook only, no app.py changes)
- [ ] **Step 2** — Modify `app.py` (8 targeted changes)
- [ ] **Step 3** — Verify `app.py` syntax with `ast.parse`
- [ ] **Step 4** — Run notebook Cell 1-3 (extract sequences) — takes ~10-30 min depending on dataset size
- [ ] **Step 5** — Run notebook Cell 4-6 (train) — ~20 epochs, fast due to tiny model
- [ ] **Step 6** — Run notebook Cell 7-8 (evaluate + save JSON)
- [ ] **Step 7** — Restart Streamlit app — it will auto-detect `bilstm_lean_dw.pt` and switch to lean pipeline

---

## Risk / Fallback

If lean pipeline accuracy is significantly lower than current BiLSTM (88.59% val):
- The 16-dim feature vector may be too sparse for the BiLSTM to learn rich patterns
- Fallback: increase feature dim to 32 by adding per-class mean/std across top-3 detections
- Or: hybrid — use YOLO 4-class probs (4-dim) + MobileNetV2 reduced projection (64-dim) = 68-dim

If `bilstm_lean_dw.pt` is deleted, app.py automatically falls back to the standard pipeline.
