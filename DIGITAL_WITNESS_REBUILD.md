# Digital Witness — Full Rebuild Plan
## Instructions for Claude Code

---

## Context & Goal

FYP system: detect shoplifting in retail CCTV footage using deep learning.

**Pipeline (lean — no MobileNetV2):**
YOLO26n (fine-tuned, 4 classes) → 16-dim feature vector → BiLSTM + Attention

**Why lean pipeline:**
- RTX 3070 local training, Streamlit cloud deployment
- MobileNetV2 adds ~800MB RAM at inference — unnecessary
- YOLO already captures rich per-frame behaviour as a compact 16-dim vector
- BiLSTM with 16-dim input: ~200K params vs ~4M — faster training, less overfitting
- Thesis framing: "optimised for low-spec edge deployment"

**Core insight from existing working repo (shoplifting_detection.py):**
Fine-tune YOLO on a labelled dataset, read per-frame class confidences directly.
We do the same but add temporal reasoning: YOLO detections → 16-dim vector → BiLSTM
classifies sequences over 7.5s windows.

---

## Dataset

### YOLO Fine-tuning
- **Name:** shopliftingvideo+handpocket Computer Vision Model (Roboflow)
- **Classes (4):** 0=normal, 1=looking-around, 2=picking-holding, 3=shoplifting
- **Format:** YOLO format — data.yaml + train/valid/test image+label folders
- **Local path:** `data/dataset/data.yaml`
- **Note:** Read class names from data.yaml at runtime — never hardcode indices

### BiLSTM Training Videos
- `D:/Santosh/Dataset/normal/` — kipshidze Normal + UCF-Crime Normal_Videos_event
- `D:/Santosh/Dataset/shoplifting/` — kipshidze Shoplifting + UCF-Crime Shoplifting

---

## Project Structure

```
Project_DigitalWitness/
├── models/
│   ├── yolo26n.pt                  # base weights (already exists)
│   ├── yolo_dw.pt                  # fine-tuned YOLO (4-class output)
│   ├── bilstm_lean_dw.pt           # trained BiLSTM
│   └── bilstm_lean_dw_info.json
├── data/
│   ├── dataset/                    # Roboflow YOLO dataset
│   │   ├── data.yaml
│   │   ├── train/images/, train/labels/
│   │   ├── valid/images/, valid/labels/
│   │   └── test/images/,  test/labels/
│   └── lean_sequences/             # .npy feature arrays per video
│       ├── normal/
│       └── shoplifting/
├── outputs/cases/
├── DigitalWitness_Pipeline.ipynb
├── app.py
├── pos_integration.py
└── run.py
```

---

## Reference Notebook Observations (train_yolov12_object_detection.ipynb)

The attached YOLOv12 training notebook shows the exact pattern to follow:

1. **Install:** `pip install ultralytics roboflow supervision`
2. **data.yaml path patch** — strip old relative paths, rewrite as absolute:
   ```python
   cfg['train'] = str(DATASET_DIR / 'train' / 'images')
   cfg['val']   = str(DATASET_DIR / 'valid' / 'images')
   cfg['test']  = str(DATASET_DIR / 'test'  / 'images')
   cfg.pop('path', None)
   ```
3. **Training call** — use `model.train()` with freeze, patience, plots
4. **Evaluation** — use `supervision.metrics.MeanAveragePrecision` exactly as shown
5. **Best weights** — copy from `runs/yolo_dw/weights/best.pt` to `models/yolo_dw.pt`

---

## Cell 2 — Full Config

```python
from pathlib import Path
import torch, yaml

ROOT        = Path('D:/Santosh/Project_DigitalWitness')
VIDEO_ROOT  = Path('D:/Santosh/Dataset')
DATASET_DIR = ROOT / 'data' / 'dataset'
SEQ_DIR     = ROOT / 'data' / 'lean_sequences'
MODELS_DIR  = ROOT / 'models'
OUTPUTS_DIR = ROOT / 'outputs' / 'cases'

for d in [SEQ_DIR/'normal', SEQ_DIR/'shoplifting', MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0 if device.type == 'cpu' else 2

# Read classes from data.yaml — never hardcode
with open(DATASET_DIR / 'data.yaml') as f:
    _cfg = yaml.safe_load(f)
YOLO_CLASSES     = _cfg['names']       # ['normal','looking-around','picking-holding','shoplifting']
BEHAVIOR_CLASSES = ['normal', 'shoplifting']

# YOLO
YOLO_BASE   = str(MODELS_DIR / 'yolo26n.pt')
YOLO_SAVE   = str(MODELS_DIR / 'yolo_dw.pt')
EPOCHS_YOLO = 50
BATCH_YOLO  = 16
FREEZE_YOLO = 10    # freeze backbone, train head only

# Sequence extraction
FPS_TARGET        = 6      # 1 frame per ~0.17s — catches sub-0.5s concealment
MAX_NORMAL_VIDEOS = 500    # cap to prevent UCF-Crime class imbalance

# BiLSTM
LEAN_FEAT_DIM  = 16
LEAN_HIDDEN    = 128
LEAN_LAYERS    = 2
LEAN_DROPOUT   = 0.3
SEQ_LEN        = 45        # 45 frames @ 6fps = 7.5s temporal window
SEQ_STRIDE     = 15
EPOCHS_BILSTM  = 20
LR_BILSTM      = 5e-4
WEIGHT_DECAY   = 1e-4
BATCH_BILSTM   = 32
PATIENCE       = 7
BILSTM_SAVE    = str(MODELS_DIR / 'bilstm_lean_dw.pt')
BILSTM_INFO    = str(MODELS_DIR / 'bilstm_lean_dw_info.json')

SMOKE_TEST = False   # True = 2-epoch check, 10 videos per class, runs in <5 min
if SMOKE_TEST:
    EPOCHS_YOLO, EPOCHS_BILSTM, MAX_NORMAL_VIDEOS = 2, 2, 10

print(f'Device: {device}  |  YOLO classes: {YOLO_CLASSES}')
```

---

## Cell 3 — YOLO Fine-tune

```python
import shutil, yaml
from ultralytics import YOLO

# Patch data.yaml to absolute paths (mirrors reference notebook)
yaml_path = DATASET_DIR / 'data.yaml'
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
cfg.update({'train': str(DATASET_DIR/'train'/'images'),
            'val':   str(DATASET_DIR/'valid'/'images'),
            'test':  str(DATASET_DIR/'test'/'images')})
cfg.pop('path', None)
with open(yaml_path, 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

model = YOLO(YOLO_BASE)
results = model.train(
    data=str(yaml_path), epochs=EPOCHS_YOLO,
    imgsz=640, batch=BATCH_YOLO, freeze=FREEZE_YOLO,
    project=str(ROOT / 'runs'), name='yolo_dw',
    patience=10, save=True, plots=True,
    device=0 if device.type == 'cuda' else 'cpu'
)
src = Path(results.save_dir) / 'weights' / 'best.pt'
shutil.copy(src, YOLO_SAVE)
print(f'[INFO] YOLO saved → {YOLO_SAVE}')
```

---

## Cell 4 — YOLO Evaluation

```python
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from ultralytics import YOLO

model = YOLO(YOLO_SAVE)
ds = sv.DetectionDataset.from_yolo(
    images_directory_path=str(DATASET_DIR / 'test' / 'images'),
    annotations_directory_path=str(DATASET_DIR / 'test' / 'labels'),
    data_yaml_path=str(DATASET_DIR / 'data.yaml')
)
preds, targets = [], []
for _, image, target in ds:
    preds.append(sv.Detections.from_ultralytics(model(image, verbose=False)[0]))
    targets.append(target)

mAP = MeanAveragePrecision().update(preds, targets).compute()
print(f'mAP50: {mAP.map50:.3f}  |  mAP50-95: {mAP.map50_95:.3f}')
mAP.plot()
```

---

## Cell 5 — Extract Feature Sequences

```python
import cv2, numpy as np, random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def extract_lean_features(results, img_w: int, img_h: int) -> np.ndarray:
    """YOLO result for one frame → 16-dim float32 feature vector."""
    feat = np.zeros(16, dtype=np.float32)
    feat[8] = feat[9] = 0.5   # default centre

    if results.boxes is None or len(results.boxes) == 0:
        return feat

    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs   = results.boxes.conf.cpu().numpy()
    xyxy    = results.boxes.xyxy.cpu().numpy()

    # Map class names → indices at runtime from YOLO_CLASSES
    name_to_idx = {n: i for i, n in enumerate(YOLO_CLASSES)}
    i_shop  = name_to_idx.get('shoplifting',    3)
    i_look  = name_to_idx.get('looking-around', 1)
    i_pick  = name_to_idx.get('picking-holding',2)
    i_norm  = name_to_idx.get('normal',         0)

    max_conf = {c: 0.0 for c in range(len(YOLO_CLASSES))}
    cnt      = {c: 0   for c in range(len(YOLO_CLASSES))}
    for c, cf in zip(cls_ids, confs):
        max_conf[c] = max(max_conf[c], float(cf))
        cnt[c] += 1

    feat[0]  = max_conf[i_shop]
    feat[1]  = max_conf[i_look]
    feat[2]  = max_conf[i_pick]
    feat[3]  = max_conf[i_norm]
    feat[4]  = min(cnt[i_shop] / 5, 1.0)
    feat[5]  = min(cnt[i_look] / 5, 1.0)
    feat[6]  = min(cnt[i_pick] / 5, 1.0)
    feat[7]  = min(len(cls_ids) / 10, 1.0)

    best    = int(np.argmax(confs))
    x1,y1,x2,y2 = xyxy[best]
    feat[8]  = ((x1+x2)/2) / img_w
    feat[9]  = ((y1+y2)/2) / img_h
    feat[10] = (x2-x1) / img_w
    feat[11] = (y2-y1) / img_h
    feat[12] = 1.0
    feat[13] = 1.0 if max_conf[i_shop] > 0 or max_conf[i_look] > 0 else 0.0
    feat[14] = 1.0 if max_conf[i_shop] > 0.5 else 0.0
    feat[15] = (cnt[i_shop]+cnt[i_look]) / max(len(cls_ids), 1)
    return feat


def process_video(yolo, video_path: Path, out_path: Path) -> bool:
    if out_path.exists(): return True
    cap     = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step    = max(1, int(src_fps / FPS_TARGET))
    w, h    = int(cap.get(3)), int(cap.get(4))
    feats, fi = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fi % step == 0:
            feats.append(extract_lean_features(yolo.predict(frame, verbose=False)[0], w, h))
        fi += 1
    cap.release()
    if len(feats) < SEQ_LEN: return False
    np.save(out_path, np.array(feats, dtype=np.float32))
    return True


yolo = YOLO(YOLO_SAVE); yolo.fuse()

for cls in BEHAVIOR_CLASSES:
    videos = list((VIDEO_ROOT / cls).glob('*.mp4')) + list((VIDEO_ROOT / cls).glob('*.avi'))
    if cls == 'normal' and MAX_NORMAL_VIDEOS:
        random.seed(42); videos = random.sample(videos, min(MAX_NORMAL_VIDEOS, len(videos)))
    ok = sum(process_video(yolo, v, SEQ_DIR/cls/(v.stem+'.npy'))
             for v in tqdm(videos, desc=cls))
    print(f'[INFO] {cls}: {ok}/{len(videos)} sequences saved')
```

---

## Cell 6 — Train LeanBiLSTM

```python
import torch.nn as nn, torch.optim as optim, json
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(1), w.squeeze(-1)


class LeanBiLSTM(nn.Module):
    def __init__(self, feat_dim=16, hidden=128, layers=2, dropout=0.3, n_cls=2):
        super().__init__()
        self.bilstm    = nn.LSTM(feat_dim, hidden, layers, batch_first=True,
                                 bidirectional=True, dropout=dropout if layers>1 else 0)
        self.attention = TemporalAttention(hidden * 2)
        self.clf       = nn.Sequential(
            nn.LayerNorm(hidden*2), nn.Dropout(dropout),
            nn.Linear(hidden*2, 64), nn.ReLU(), nn.Linear(64, n_cls)
        )
    def forward(self, x):
        out, _ = self.bilstm(x)
        ctx, w = self.attention(out)
        return self.clf(ctx), w


class LeanSeqDataset(Dataset):
    def __init__(self, seq_dir, classes, seq_len, stride):
        self.items = []
        for label, cls in enumerate(classes):
            for fp in Path(seq_dir, cls).glob('*.npy'):
                arr = np.load(fp)
                for s in range(0, len(arr)-seq_len+1, stride):
                    self.items.append((arr[s:s+seq_len], label))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        a, l = self.items[i]; return torch.FloatTensor(a), l


ds     = LeanSeqDataset(SEQ_DIR, BEHAVIOR_CLASSES, SEQ_LEN, SEQ_STRIDE)
labels = [it[1] for it in ds.items]
tr_i, va_i = train_test_split(range(len(ds)), test_size=0.2, stratify=labels, random_state=42)

counts  = [labels.count(0), labels.count(1)]
w_samp  = WeightedRandomSampler([1/counts[labels[i]] for i in tr_i], len(tr_i), True)
tr_dl   = DataLoader(Subset(ds, tr_i), BATCH_BILSTM, sampler=w_samp, num_workers=NUM_WORKERS)
va_dl   = DataLoader(Subset(ds, va_i), BATCH_BILSTM, shuffle=False,  num_workers=NUM_WORKERS)

model   = LeanBiLSTM(LEAN_FEAT_DIM, LEAN_HIDDEN, LEAN_LAYERS, LEAN_DROPOUT).to(device)
cw      = torch.FloatTensor([max(counts)/c for c in counts]).to(device)
crit    = nn.CrossEntropyLoss(weight=cw)
opt     = optim.AdamW(model.parameters(), lr=LR_BILSTM, weight_decay=WEIGHT_DECAY)
sched   = optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

best, no_imp = 0.0, 0
for ep in range(1, EPOCHS_BILSTM+1):
    model.train(); tl=tc=tt=0
    for x,y in tr_dl:
        x,y = x.to(device),y.to(device); opt.zero_grad()
        lg,_ = model(x); loss = crit(lg,y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl+=loss.item()*len(x); tc+=(lg.argmax(1)==y).sum().item(); tt+=len(x)

    model.eval(); vl=vc=vt=0
    with torch.no_grad():
        for x,y in va_dl:
            x,y=x.to(device),y.to(device); lg,_=model(x)
            vl+=crit(lg,y).item()*len(x); vc+=(lg.argmax(1)==y).sum().item(); vt+=len(x)

    ta,va = tc/tt, vc/vt
    print(f'Ep {ep:3d} | train={ta:.3f} val={va:.3f} lr={sched.get_last_lr()[0]:.1e}')
    sched.step()
    if va > best:
        best=va; torch.save(model.state_dict(), BILSTM_SAVE); no_imp=0; print(f'  ✓ saved')
    else:
        no_imp+=1
        if no_imp>=PATIENCE: print(f'Early stop ep {ep}'); break

json.dump({'pipeline':'lean_b','feat_dim':LEAN_FEAT_DIM,'hidden':LEAN_HIDDEN,
           'seq_len':SEQ_LEN,'best_val_acc':best,'classes':BEHAVIOR_CLASSES},
          open(BILSTM_INFO,'w'), indent=2)
print(f'[INFO] Best val accuracy: {best:.3f}')
```

---

## Cell 7 — Evaluate BiLSTM

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model.load_state_dict(torch.load(BILSTM_SAVE, map_location=device))
model.eval()
all_p, all_y = [], []
with torch.no_grad():
    for x, y in va_dl:
        lg, _ = model(x.to(device))
        all_p.extend(lg.argmax(1).cpu().tolist())
        all_y.extend(y.tolist())

print(classification_report(all_y, all_p, target_names=BEHAVIOR_CLASSES))

cm = confusion_matrix(all_y, all_p)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(cm, display_labels=BEHAVIOR_CLASSES).plot(ax=ax, cmap='Blues')
plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / 'lean_confusion.png'), dpi=150)
plt.show()
```

---

## Cell 8 — End-to-End Smoke Test

```python
# Test full inference on one video file
TEST_VIDEO = str(VIDEO_ROOT / 'shoplifting' / list((VIDEO_ROOT/'shoplifting').glob('*.mp4'))[0].name)

yolo  = YOLO(YOLO_SAVE); yolo.fuse()
model = LeanBiLSTM(LEAN_FEAT_DIM, LEAN_HIDDEN, LEAN_LAYERS, LEAN_DROPOUT).to(device)
model.load_state_dict(torch.load(BILSTM_SAVE, map_location=device)); model.eval()

cap     = cv2.VideoCapture(TEST_VIDEO)
fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
step    = max(1, int(fps_src / FPS_TARGET))
w, h    = int(cap.get(3)), int(cap.get(4))
feats, fi = [], 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if fi % step == 0:
        feats.append(extract_lean_features(yolo.predict(frame, verbose=False)[0], w, h))
    fi += 1
cap.release()

if len(feats) >= SEQ_LEN:
    seq   = torch.FloatTensor(feats[:SEQ_LEN]).unsqueeze(0).to(device)
    with torch.no_grad(): logits, attn = model(seq)
    probs = torch.softmax(logits, dim=1)[0]
    pred  = BEHAVIOR_CLASSES[logits.argmax(1).item()]
    print(f'[RESULT] {pred.upper()} — normal:{probs[0]:.2%}  shoplifting:{probs[1]:.2%}')
    print(f'[ATTN]   Top 3 frames: {attn[0].topk(3).indices.tolist()}')
else:
    print(f'[WARN] Video too short: only {len(feats)} frames extracted')
```

---

## app.py Changes (6 targeted edits only — do not rewrite the whole file)

### Edit 1 — Add after existing LSTM_PATH line
```python
BILSTM_LEAN_PATH = MODELS_DIR / "bilstm_lean_dw.pt"
LEAN_FEAT_DIM    = 16
LEAN_HIDDEN      = 128
USE_LEAN         = BILSTM_LEAN_PATH.exists()
```

### Edit 2 — Add LeanBiLSTM + TemporalAttention class definitions
Copy the two classes from Cell 6 verbatim into app.py (after imports section).

### Edit 3 — `_build_lean_model()` helper
```python
def _build_lean_model():
    m = LeanBiLSTM(LEAN_FEAT_DIM, LEAN_HIDDEN)
    m.load_state_dict(torch.load(BILSTM_LEAN_PATH, map_location=_get_device()))
    return m.eval()
```

### Edit 4 — In `run_pipeline()`, replace MobileNetV2 loading block
```python
if USE_LEAN:
    lstm_model = _build_lean_model().to(dev)
else:
    combined_model = _build_combined_model(...).to(dev)
    # existing code unchanged
```

### Edit 5 — In video processing loop, replace feature extraction
```python
if USE_LEAN:
    feat = extract_lean_features(results, width, height)   # reuse Cell 5 function
    all_feats.append(feat)
else:
    # existing MobileNetV2 extract_cnn_feat call unchanged
```

### Edit 6 — Feature dimension for padding
```python
_fdim     = LEAN_FEAT_DIM if USE_LEAN else MOBILENET_DIM
feats_arr = np.array(all_feats) if all_feats else np.zeros((1, _fdim))
```

### DO NOT change
- POS audit tab (product_pickup_events still populated from YOLO boxes)
- XAI attention weight display
- Bias/fairness scores
- Alert generation
- All CSS and layout

---

## Code Quality (examiner will read this)

1. Max 60 lines per function
2. `print(f'[INFO] ...')` style only — no bare print statements
3. No unused imports
4. Type hints on function signatures
5. Each notebook cell max 80 lines
6. `SMOKE_TEST = True` must run the entire notebook end-to-end in under 5 minutes
7. Comments explain WHY, not WHAT

---

## Execution Order

1. Confirm `data/dataset/data.yaml` exists and lists 4 classes
2. Notebook Cell 1 — GPU check (RTX 3070 must show)
3. Notebook Cell 2 — config (run with `SMOKE_TEST=True` first)
4. Notebook Cell 3 — YOLO fine-tune (~30 min, ~2 min smoke)
5. Notebook Cell 4 — YOLO eval — **stop if mAP50 < 0.70, check dataset**
6. Notebook Cell 5 — extract sequences (~20-60 min depending on video count)
7. Notebook Cell 6 — train BiLSTM (~15 min)
8. Notebook Cell 7 — evaluate — check shoplifting recall > 0.75
9. Notebook Cell 8 — smoke test on one video
10. `python run.py` → upload video → verify result in browser

---

## Expected Metrics

| Model | Minimum |
|-------|---------|
| YOLO mAP50 | > 0.75 |
| BiLSTM val accuracy | > 0.82 |
| BiLSTM shoplifting recall | > 0.75 |

**If metrics low:** Most likely cause is too few training sequences (< 100 per class).
Print sequence counts after Cell 5 before adjusting hyperparameters.

---

## Files to Create

In this order:
1. `DigitalWitness_Pipeline.ipynb` — 8 cells per spec above
2. `app.py` — 6 targeted edits to existing file
3. `pos_integration.py` — copy unchanged
4. `run.py` — copy unchanged

## Files NOT to Create

- No `train.py`
- No `lean_pipeline.py`
- No `mobilenet_extractor.py`
- No changes to `datagetimage.py`
