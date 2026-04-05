# Digital Witness — Final Implementation Plan
## YOLO26n + EfficientNetV2-S + BiLSTM + Temporal Attention

---

## Research Positioning

This architecture fills an identified gap in the literature:

| System | Detection | CNN Backbone | Temporal | Tracker |
|--------|-----------|-------------|---------|---------|
| Most papers | YOLOv5 | InceptionV3 | LSTM | DeepSORT |
| Best existing | YOLOv8 | EfficientNetV2-B0 | None (frame-level) | ByteTrack |
| **This thesis** | **YOLO26n** | **EfficientNetV2-S** | **BiLSTM+Attention** | **ByteTrack** |

Contribution: First known system combining YOLO26n (NMS-free, 2026) with
EfficientNetV2-S as a temporal feature extractor feeding a BiLSTM with
attention-based XAI output for shoplifting detection.

---

## Full Pipeline

```
Video frame
    │
    ▼
YOLO26n (fine-tuned, 4 classes)
+ ByteTrack (person ID across frames)
    │
    ├─ per-person bounding box crop
    │
    ▼
EfficientNetV2-S (frozen backbone, last 2 blocks unfrozen)
    │  1280-dim spatial feature per frame
    │
    ▼
Sliding window queue (45 frames × 1280-dim per tracked person)
    │
    ▼
BiLSTM (256 hidden, 2 layers, bidirectional)
    │
    ▼
Temporal Attention (learns which frames matter most → XAI)
    │
    ▼
Classifier → normal / shoplifting
    │
    ├─ Intent score (behaviour + duration)
    ├─ Attention weight map (XAI output)
    ├─ Bias-aware adjustment
    └─ Alert + Case file JSON
```

---

## Datasets

### YOLO Fine-tuning (behaviour detection per frame)
- Dataset: shopliftingvideo+handpocket (Roboflow)
- Classes (4): normal, looking-around, picking-holding, shoplifting
- Format: YOLO, data.yaml + train/valid/test folders
- Path: `data/dataset/data.yaml`
- Rule: always read class names from data.yaml at runtime

### BiLSTM Video Training
- Normal: `D:/Santosh/Dataset/normal/`
- Shoplifting: `D:/Santosh/Dataset/shoplifting/`
- Sources: kipshidze Kaggle + UCF-Crime

---

## Project Structure

```
Project_DigitalWitness/
├── models/
│   ├── yolo26n.pt                  # base weights (exists)
│   ├── yolo_dw.pt                  # fine-tuned YOLO (4-class)
│   ├── efficientnet_dw.pt          # fine-tuned EfficientNetV2-S
│   ├── bilstm_dw.pt                # trained BiLSTM
│   ├── efficientnet_dw_info.json
│   └── bilstm_dw_info.json
├── data/
│   ├── dataset/                    # Roboflow YOLO dataset
│   └── frames/                     # extracted frames
│       ├── normal/
│       └── shoplifting/
├── outputs/cases/
├── DigitalWitness_Pipeline.ipynb   # 8-cell training notebook
├── app.py                          # Streamlit dashboard
├── pos_integration.py
└── run.py
```

---

## Notebook — 8 Cells

---

### CELL 1 — Install + Environment Check

```python
import subprocess, sys, torch, platform

pkgs = [
    'ultralytics>=8.3.0', 'supervision',
    'opencv-python', 'numpy', 'pandas',
    'matplotlib', 'scikit-learn', 'Pillow', 'tqdm', 'pyyaml'
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *pkgs, '-q'])

print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}', end='')
if torch.cuda.is_available():
    print(f' — {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print()
```

---

### CELL 2 — Configuration (all hyperparameters in one place)

```python
import torch, yaml, platform
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT        = Path('D:/Santosh/Project_DigitalWitness')
VIDEO_ROOT  = Path('D:/Santosh/Dataset')
DATASET_DIR = ROOT / 'data' / 'dataset'
FRAMES_DIR  = ROOT / 'data' / 'frames'
MODELS_DIR  = ROOT / 'models'
OUTPUTS_DIR = ROOT / 'outputs' / 'cases'

for d in [FRAMES_DIR/'normal', FRAMES_DIR/'shoplifting', MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
PIN_MEMORY  = device.type == 'cuda'

# ── Classes from data.yaml (never hardcode) ───────────────────────
with open(DATASET_DIR / 'data.yaml') as f:
    _cfg = yaml.safe_load(f)
YOLO_CLASSES     = _cfg['names']   # ['normal','looking-around','picking-holding','shoplifting']
BEHAVIOR_CLASSES = ['normal', 'shoplifting']

# ── YOLO ──────────────────────────────────────────────────────────
YOLO_BASE   = str(MODELS_DIR / 'yolo26n.pt')
YOLO_SAVE   = str(MODELS_DIR / 'yolo_dw.pt')
EPOCHS_YOLO = 50
BATCH_YOLO  = 16
FREEZE_YOLO = 10   # freeze backbone, train detection head only

# ── Frame extraction ──────────────────────────────────────────────
FPS_TARGET        = 6    # catches concealment events as short as 0.17s
MAX_NORMAL_VIDEOS = 500  # cap UCF-Crime imbalance

# ── EfficientNetV2-S ──────────────────────────────────────────────
EFFNET_SAVE     = str(MODELS_DIR / 'efficientnet_dw.pt')
EFFNET_INFO     = str(MODELS_DIR / 'efficientnet_dw_info.json')
FEAT_DIM        = 1280
EPOCHS_EFFNET   = 25
LR_EFFNET       = 1e-4
WD_EFFNET       = 1e-4
BATCH_EFFNET    = 32 if device.type == 'cuda' else 16

# ── BiLSTM ────────────────────────────────────────────────────────
BILSTM_SAVE   = str(MODELS_DIR / 'bilstm_dw.pt')
BILSTM_INFO   = str(MODELS_DIR / 'bilstm_dw_info.json')
SEQ_LEN       = 45    # 45 frames @ 6fps = 7.5s temporal window
SEQ_STRIDE    = 15
HIDDEN        = 256
LAYERS        = 2
DROPOUT       = 0.3
EPOCHS_BILSTM = 20
LR_BILSTM     = 5e-4
WD_BILSTM     = 1e-4
BATCH_BILSTM  = 16
PATIENCE      = 7

# ── Smoke test ────────────────────────────────────────────────────
SMOKE_TEST = False   # set True for 2-epoch pipeline verification
if SMOKE_TEST:
    EPOCHS_YOLO = EPOCHS_EFFNET = EPOCHS_BILSTM = 2
    MAX_NORMAL_VIDEOS = 10

print(f'Device       : {device}')
print(f'YOLO classes : {YOLO_CLASSES}')
print(f'Smoke test   : {SMOKE_TEST}')
```

---

### CELL 3 — YOLO26n Fine-tuning

```python
import shutil, yaml
from pathlib import Path
from ultralytics import YOLO

# Patch data.yaml to absolute paths
yaml_path = DATASET_DIR / 'data.yaml'
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
cfg.update({
    'train': str(DATASET_DIR / 'train' / 'images'),
    'val':   str(DATASET_DIR / 'valid' / 'images'),
    'test':  str(DATASET_DIR / 'test'  / 'images'),
})
cfg.pop('path', None)
with open(yaml_path, 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

model   = YOLO(YOLO_BASE)
results = model.train(
    data=str(yaml_path), epochs=EPOCHS_YOLO,
    imgsz=640, batch=BATCH_YOLO, freeze=FREEZE_YOLO,
    project=str(ROOT / 'runs'), name='yolo_dw',
    patience=10, save=True, plots=True,
    device=0 if device.type == 'cuda' else 'cpu',
    workers=NUM_WORKERS,
)

src = Path(results.save_dir) / 'weights' / 'best.pt'
shutil.copy(src, YOLO_SAVE)

# Print key metrics
m     = results.results_dict
map50 = m.get('metrics/mAP50(B)', m.get('metrics/mAP_0.5', 0))
print(f'[INFO] YOLO saved → {YOLO_SAVE}')
print(f'[INFO] mAP50: {map50:.3f}')
```

**Stop here if mAP50 < 0.70 — check dataset before continuing.**

---

### CELL 4 — EfficientNetV2-S Training

```python
import cv2, random, json
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# Step 4a — Frame extraction from videos
def extract_frames(video_root, out_dir, classes, fps_target):
    for cls in classes:
        dst = Path(out_dir) / cls
        dst.mkdir(parents=True, exist_ok=True)
        existing = list(dst.glob('*.jpg'))
        if existing:
            print(f'  [{cls}] {len(existing)} frames exist — skipping')
            continue

        videos = (list((Path(video_root) / cls).glob('*.mp4')) +
                  list((Path(video_root) / cls).glob('*.avi')))
        if not videos:
            print(f'  [{cls}] WARNING: no videos in {Path(video_root)/cls}')
            continue

        if cls == 'normal' and MAX_NORMAL_VIDEOS and len(videos) > MAX_NORMAL_VIDEOS:
            random.seed(42)
            videos = random.sample(videos, MAX_NORMAL_VIDEOS)

        count = 0
        for vid in tqdm(videos, desc=f'  {cls}', leave=False):
            cap     = cv2.VideoCapture(str(vid))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            step    = max(1, int(src_fps / fps_target))
            fi      = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if fi % step == 0:
                    cv2.imwrite(str(dst / f'{vid.stem}_f{fi:06d}.jpg'),
                                cv2.resize(frame, (224, 224)))
                    count += 1
                fi += 1
            cap.release()
        print(f'  [{cls}] {count} frames extracted')


print('[INFO] Extracting frames...')
extract_frames(VIDEO_ROOT, FRAMES_DIR, BEHAVIOR_CLASSES, FPS_TARGET)
for cls in BEHAVIOR_CLASSES:
    print(f'  {cls}: {len(list((FRAMES_DIR/cls).glob("*.jpg")))} frames')


# Step 4b — Dataset + DataLoaders
class FrameDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        return self.transform(img) if self.transform else img, label


all_samples = [(str(p), i)
               for i, cls in enumerate(BEHAVIOR_CLASSES)
               for p in (FRAMES_DIR / cls).glob('*.jpg')]
labels = [l for _, l in all_samples]
train_s, val_s = train_test_split(all_samples, test_size=0.2, stratify=labels, random_state=42)

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cls_counts  = [labels.count(0), labels.count(1)]
sample_w    = [1.0 / cls_counts[l] for _, l in train_s]
train_dl    = DataLoader(FrameDataset(train_s, train_tf), BATCH_EFFNET,
                         sampler=WeightedRandomSampler(sample_w, len(train_s)),
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_dl      = DataLoader(FrameDataset(val_s, val_tf), BATCH_EFFNET,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print(f'[INFO] Train: {len(train_s)}  Val: {len(val_s)}')


# Step 4c — Model (unfreeze last 2 feature blocks)
effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
for p in effnet.parameters(): p.requires_grad = False
for layer in list(effnet.features.children())[-2:]:
    for p in layer.parameters(): p.requires_grad = True
effnet.classifier = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(1280, 512),
    nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, len(BEHAVIOR_CLASSES))
)
effnet = effnet.to(device)
print(f'[INFO] Trainable params: {sum(p.numel() for p in effnet.parameters() if p.requires_grad):,}')


# Step 4d — Training loop
cw        = torch.FloatTensor([max(cls_counts)/c for c in cls_counts]).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, effnet.parameters()),
                       lr=LR_EFFNET, weight_decay=WD_EFFNET)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
best_acc, no_imp = 0.0, 0

for epoch in range(1, EPOCHS_EFFNET + 1):
    effnet.train(); tl = tc = tt = 0
    for imgs, lbls in tqdm(train_dl, desc=f'Ep {epoch}/{EPOCHS_EFFNET}', leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = effnet(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        tl += loss.item()*len(imgs); tc += (out.argmax(1)==lbls).sum().item(); tt += len(imgs)

    effnet.eval(); vl = vc = vt = 0
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = effnet(imgs)
            vl += criterion(out, lbls).item()*len(imgs)
            vc += (out.argmax(1)==lbls).sum().item(); vt += len(imgs)

    ta, va = tc/tt, vc/vt
    print(f'Epoch {epoch:3d} | train={ta:.3f} val={va:.3f} lr={optimizer.param_groups[0]["lr"]:.1e}')
    scheduler.step(vl/vt)

    if va > best_acc:
        best_acc = va
        torch.save(effnet.state_dict(), EFFNET_SAVE)
        no_imp = 0; print(f'  ✓ val_acc={va:.3f}')
    else:
        no_imp += 1
        if no_imp >= PATIENCE: print(f'  Early stop epoch {epoch}'); break

json.dump({'backbone':'efficientnet_v2_s','feat_dim':FEAT_DIM,
           'best_val_acc':best_acc,'classes':BEHAVIOR_CLASSES},
          open(EFFNET_INFO,'w'), indent=2)
print(f'[INFO] EfficientNetV2-S best val accuracy: {best_acc:.3f}')
```

---

### CELL 5 — BiLSTM + Temporal Attention Training

```python
import json
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.model_selection import train_test_split
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):               # x: (B, T, dim)
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(1), w.squeeze(-1)


class CNNBiLSTMAttention(nn.Module):
    """EfficientNetV2-S → BiLSTM → Temporal Attention → classifier."""

    def __init__(self, num_classes=2, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.cnn       = base.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        for p in self.cnn.parameters(): p.requires_grad = False

        self.bilstm    = nn.LSTM(FEAT_DIM, hidden, layers, batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Dropout(dropout),
            nn.Linear(hidden * 2, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def extract_feat(self, x):          # x: (1, 3, 224, 224) — inference helper
        with torch.no_grad():
            return self.pool(self.cnn(x)).flatten(1)   # (1, 1280)

    def forward(self, x):               # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape
        f = self.pool(self.cnn(x.view(B*T, C, H, W))).view(B, T, -1)
        out, _ = self.bilstm(f)
        ctx, w  = self.attention(out)
        return self.classifier(ctx), w


# Sequence dataset — NO shuffle inside __init__
seq_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SequenceDataset(Dataset):
    def __init__(self, frame_root, classes, seq_len, transform=None):
        self.seq_len   = seq_len
        self.transform = transform
        self.sequences = []
        for label, cls in enumerate(classes):
            vid_frames = defaultdict(list)
            for p in sorted((Path(frame_root) / cls).glob('*.jpg')):
                vid_frames['_'.join(p.stem.split('_')[:-1])].append(p)
            for paths in vid_frames.values():
                paths = sorted(paths)
                for s in range(0, len(paths) - seq_len + 1, seq_len):
                    chunk = paths[s:s+seq_len]
                    if len(chunk) == seq_len:
                        self.sequences.append((chunk, label))

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        paths, label = self.sequences[idx]
        frames = [self.transform(Image.open(p).convert('RGB')) for p in paths]
        return torch.stack(frames), label


print('[INFO] Building sequence dataset...')
full_ds    = SequenceDataset(FRAMES_DIR, BEHAVIOR_CLASSES, SEQ_LEN, seq_tf)
seq_labels = [s[1] for s in full_ds.sequences]
print(f'[INFO] Sequences — total: {len(full_ds)} | normal: {seq_labels.count(0)} | shoplifting: {seq_labels.count(1)}')

if len(full_ds) < 4:
    raise RuntimeError('Too few sequences — check frame extraction completed correctly')

tr_i, va_i = train_test_split(range(len(full_ds)), test_size=0.2,
                               stratify=seq_labels, random_state=42)
cls_counts = [seq_labels.count(0), seq_labels.count(1)]
sw = [1.0/cls_counts[seq_labels[i]] for i in tr_i]
tr_dl = DataLoader(Subset(full_ds, tr_i), BATCH_BILSTM,
                   sampler=WeightedRandomSampler(sw, len(tr_i)),
                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
va_dl = DataLoader(Subset(full_ds, va_i), BATCH_BILSTM,
                   shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

model     = CNNBiLSTMAttention(len(BEHAVIOR_CLASSES)).to(device)
cw        = torch.FloatTensor([max(cls_counts)/c for c in cls_counts]).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_BILSTM, weight_decay=WD_BILSTM)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
best_acc, no_imp = 0.0, 0

for epoch in range(1, EPOCHS_BILSTM + 1):
    model.train(); tl = tc = tt = 0
    for seqs, lbls in tqdm(tr_dl, desc=f'BiLSTM Ep {epoch}/{EPOCHS_BILSTM}', leave=False):
        seqs, lbls = seqs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits, _ = model(seqs)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()*len(seqs); tc += (logits.argmax(1)==lbls).sum().item(); tt += len(seqs)

    model.eval(); vl = vc = vt = 0
    with torch.no_grad():
        for seqs, lbls in va_dl:
            seqs, lbls = seqs.to(device), lbls.to(device)
            logits, _ = model(seqs)
            vl += criterion(logits, lbls).item()*len(seqs)
            vc += (logits.argmax(1)==lbls).sum().item(); vt += len(seqs)

    ta, va = tc/tt, vc/vt
    print(f'Epoch {epoch:3d}/{EPOCHS_BILSTM} | train={ta:.3f} val={va:.3f}')
    scheduler.step(vl/vt)

    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), BILSTM_SAVE)
        no_imp = 0; print(f'  ✓ val_acc={va:.3f}')
    else:
        no_imp += 1
        if no_imp >= PATIENCE: print(f'  Early stop epoch {epoch}'); break

json.dump({'backbone':'efficientnet_v2_s','best_val_acc':best_acc,
           'seq_len':SEQ_LEN,'hidden':HIDDEN,'layers':LAYERS,
           'feat_dim':FEAT_DIM,'classes':BEHAVIOR_CLASSES},
          open(BILSTM_INFO,'w'), indent=2)
print(f'[INFO] BiLSTM best val accuracy: {best_acc:.3f}')
```

---

### CELL 6 — Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model.load_state_dict(torch.load(BILSTM_SAVE, map_location=device))
model.eval()
all_p, all_y = [], []
with torch.no_grad():
    for seqs, lbls in va_dl:
        logits, _ = model(seqs.to(device))
        all_p.extend(logits.argmax(1).cpu().tolist())
        all_y.extend(lbls.tolist())

print(classification_report(all_y, all_p, target_names=BEHAVIOR_CLASSES))

fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix(all_y, all_p),
                       display_labels=BEHAVIOR_CLASSES).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix — Digital Witness')
plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / 'confusion_matrix.png'), dpi=150)
plt.show()
```

---

### CELL 7 — POS Integration Setup

```python
import sys
from datetime import datetime
sys.path.insert(0, str(ROOT))
from pos_integration import MockPOSDatabase

pos_db = MockPOSDatabase()
pos_db.generate_sessions(n=20)
pos_db.add_suspicious_session(timestamp=datetime.now(), items_billed=2, items_detected=5)
pos_db.save(str(OUTPUTS_DIR / 'mock_pos_transactions.json'))
pos_db.summary()
```

---

### CELL 8 — End-to-End Smoke Test

```python
import cv2
from torchvision import transforms
from ultralytics import YOLO as YOLOModel

test_vid = next((VIDEO_ROOT / 'shoplifting').glob('*.mp4'), None)
assert test_vid is not None, 'No shoplifting test video found'
print(f'[INFO] Testing: {test_vid.name}')

yolo  = YOLOModel(YOLO_SAVE)
model.load_state_dict(torch.load(BILSTM_SAVE, map_location=device))
model.eval()

inf_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cap     = cv2.VideoCapture(str(test_vid))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
step    = max(1, int(src_fps / FPS_TARGET))
feats, fi = [], 0

while len(feats) < SEQ_LEN:
    ret, frame = cap.read()
    if not ret: break
    if fi % step == 0:
        t = inf_tf(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        feats.append(model.extract_feat(t).squeeze(0).cpu())
    fi += 1
cap.release()

if len(feats) < SEQ_LEN:
    print(f'[WARN] Video too short — only {len(feats)} frames extracted')
else:
    seq = torch.stack(feats).unsqueeze(0).to(device)   # (1, 45, 1280)
    with torch.no_grad():
        out, _    = model.bilstm(seq)
        ctx, attn = model.attention(out)
        logits    = model.classifier(ctx)
    probs = torch.softmax(logits, dim=1)[0]
    pred  = BEHAVIOR_CLASSES[logits.argmax(1).item()]
    print(f'[RESULT] {pred.upper()}')
    print(f'  normal={probs[0]:.2%}  shoplifting={probs[1]:.2%}')
    print(f'  Top attention frames: {attn[0].topk(3).indices.tolist()}')
```

---

## app.py — Targeted Changes Only

### 1. Model path constants (add after existing LSTM_PATH)
```python
EFFNET_PATH  = MODELS_DIR / "efficientnet_dw.pt"
BILSTM_PATH  = MODELS_DIR / "bilstm_dw.pt"
FEAT_DIM     = 1280
```

### 2. Replace _build_combined_model() backbone
```python
# Change:
from torchvision.models import MobileNet_V2_Weights
base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
self.cnn = base.features

# To:
from torchvision.models import EfficientNet_V2_S_Weights
base = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
self.cnn  = base.features
self.pool = nn.AdaptiveAvgPool2d(1)  # EfficientNetV2 needs explicit pool
```

### 3. Update extract_cnn_feat
```python
def extract_cnn_feat(self, x):
    with torch.no_grad():
        f = self.cnn(x)
        return self.pool(f).flatten(1)   # (1, 1280)
```

### 4. DO NOT change
- POS audit tab
- XAI attention weight display
- Bias/fairness scores
- Alert generation
- All CSS and layout

---

## Code Quality Rules (examiner sees this code)

1. Max 60 lines per function — split if longer
2. Only `print(f'[INFO] ...')` style logs
3. No unused imports
4. Type hints on function signatures
5. Each notebook cell max 80 lines
6. SMOKE_TEST=True must finish the whole notebook in under 5 minutes
7. Comments explain WHY, not WHAT

---

## Execution Order

1. Verify `data/dataset/data.yaml` has 4 classes
2. Cell 1 — check CUDA shows RTX 3070
3. Cell 2 — set SMOKE_TEST=True, run to verify no import errors
4. Cell 3 — YOLO fine-tune (~30 min / ~2 min smoke)
5. **Stop if mAP50 < 0.70**
6. Cell 4 — EfficientNetV2-S train (~20 min)
7. **Stop if val_acc < 0.80**
8. Cell 5 — BiLSTM train (~15 min)
9. Cell 6 — evaluate, check shoplifting recall > 0.75
10. Cell 7 — POS setup
11. Cell 8 — smoke test on one video
12. Update app.py (4 targeted edits)
13. python run.py → test in browser

---

## Expected Metrics

| Model | Minimum target |
|-------|---------------|
| YOLO mAP50 | > 0.75 |
| EfficientNetV2-S val accuracy | > 0.88 |
| BiLSTM val accuracy | > 0.85 |
| BiLSTM shoplifting recall | > 0.75 |

---

## Files to Create

1. `DigitalWitness_Pipeline.ipynb` — 8 cells exactly as above
2. `app.py` — 4 targeted edits to existing file
3. `pos_integration.py` — copy unchanged
4. `run.py` — copy unchanged

## Files NOT to Create

- No train.py
- No separate feature_extractor.py
- No lean_pipeline.py
