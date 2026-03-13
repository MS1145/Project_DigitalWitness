# YOLO26 Fine-Tuning Plan & Domain Shift Observation

## Observation: Domain Shift in `yolo26_retail.pt`

### What happened
`yolo26_retail.pt` was fine-tuned on the **shoplifting-annotation** Roboflow dataset
(`data/dataset/` — 578 train + 116 val images). This dataset contains annotated frames
from a specific in-store surveillance camera with 24 retail-specific classes
(e.g. `person-with-filled-basket-trolly`, `person-with-carrying-item`).

When the same model is run on frames from the **UCF-Crime / Kipshidze behaviour
dataset** (`frames/shoplifting/`, `frames/normal/` — 31,302 frames total), it returns
**zero detections at any confidence threshold**, including 0.05.

### Root cause
The two datasets come from completely different camera domains:

| Property | Retail annotation dataset | Behaviour dataset (UCF-Crime/Kipshidze) |
|---|---|---|
| Source | Single in-store CCTV | UCF-Crime + Kipshidze surveillance |
| Frame style | High-angle, static store scene | Variable angles, scene movement |
| Lighting | Consistent indoor retail | Mixed indoor/outdoor, low-light |
| Image count | 694 annotated frames | 31,302 frames (no YOLO labels) |
| Classes | 24 composite retail states | Not annotated for YOLO |

The fine-tuned model learned the visual characteristics of the retail store images
so strongly that it cannot generalise to the UCF-Crime footage. This is a classic
**domain shift / distribution mismatch** problem.

### Current workaround
`app.py` now uses `yolo26n.pt` (base COCO model) as the primary detection model.
It reliably detects the `person` class across surveillance footage domains.
`yolo26_retail.pt` is kept as fallback only.

---

## Fine-Tuning Strategy

### Option 1 — Auto-label behaviour frames and add to YOLO dataset (Recommended)

Use `yolo26n.pt` (which works well) to auto-generate `person` bounding box labels for
a subset of the behaviour frames, then combine with the existing retail annotations and
re-fine-tune YOLO26.

**Steps:**
```python
# Run in notebook or script
from ultralytics import YOLO
from pathlib import Path
import shutil, random

base = YOLO('models/yolo26n.pt')

# Sample frames from both behaviour classes
frames_dir = Path('frames')
out_img = Path('data/dataset_v2/train/images')
out_lbl = Path('data/dataset_v2/train/labels')
out_img.mkdir(parents=True, exist_ok=True)
out_lbl.mkdir(parents=True, exist_ok=True)

all_frames = list((frames_dir / 'shoplifting').glob('*.jpg')) + \
             list((frames_dir / 'normal').glob('*.jpg'))
sample = random.sample(all_frames, 1000)  # start with 1000 frames

for img_path in sample:
    res = base(img_path, conf=0.3, verbose=False)[0]
    lines = []
    for box, cls_id in zip(res.boxes.xyxyn.cpu().numpy(),
                           res.boxes.cls.cpu().numpy()):
        if int(cls_id) == 0:  # person only
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w  = box[2] - box[0]
            h  = box[3] - box[1]
            lines.append(f"9 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # class 9 = person
    if lines:
        shutil.copy(img_path, out_img / img_path.name)
        (out_lbl / img_path.with_suffix('.txt').name).write_text('\n'.join(lines))
```

Then copy the existing `data/dataset/train/` images+labels into `data/dataset_v2/train/`
and fine-tune:

```python
model = YOLO('models/yolo26n.pt')  # start from base, not retail model
model.train(
    data='data/dataset_v2/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=1e-4,
    name='yolo26_dw_v2',
    project='runs/detect',
    patience=10,
)
```

**Expected outcome:** YOLO detects both retail composite states (from original
annotations) AND plain persons in surveillance footage (from auto-labels).

---

### Option 2 — Two-model inference (Current + Retail)

Keep `yolo26n.pt` for person tracking (it already works) and run `yolo26_retail.pt`
in parallel on each frame solely for product/state detection (basket, handbag, etc.).
Person bounding boxes come from the base model; product/state labels come from the
retail model.

This avoids any re-training but adds inference overhead (~2× YOLO calls per frame).

```python
# In run_pipeline():
person_model  = YOLO('models/yolo26n.pt')      # reliable person detection
product_model = YOLO('models/yolo26_retail.pt') # retail product states

# Per frame:
person_res  = person_model.track(frame, ...)
product_res = product_model(frame, conf=0.2, verbose=False)[0]
```

---

### Option 3 — Full joint training (all 3 models)

Re-train all three models together using the behaviour video frames with both
YOLO annotations and MobileNetV2/BiLSTM labels. This requires:

1. **YOLO labels** — auto-generated as in Option 1 above
2. **MobileNetV2 / BiLSTM labels** — already exist (`frames/normal/` and `frames/shoplifting/`)
3. Train YOLO26 fine-tune first (Option 1)
4. Re-run frame extraction using the NEW fine-tuned YOLO to get tighter person crops
5. Re-train MobileNetV2 on those person crops (instead of full frames)
6. Re-train BiLSTM on the new MobileNetV2 features

**This is the most rigorous approach** and would give the best end-to-end performance,
but requires the most time.

---

## Dataset Summary

| Dataset | Location | Size | Use |
|---|---|---|---|
| YOLO retail annotations | `data/dataset/` | 578 train + 116 val images | Current YOLO fine-tune source |
| Behaviour frames | `frames/normal/` + `frames/shoplifting/` | 10,730 + 20,572 frames | MobileNetV2 + BiLSTM training |
| Auto-labelled frames (to create) | `data/dataset_v2/` | ~1,000+ frames | YOLO v2 fine-tune |

## Recommended Action

1. Run **Option 1** to generate `data/dataset_v2/` with auto-labelled person boxes
2. Fine-tune YOLO26 on the merged dataset (retail annotations + auto-labelled surveillance frames)
3. Replace `models/yolo26n.pt` usage in `app.py` with the new `yolo26_dw_v2.pt`
4. Optionally follow **Option 3** to re-train MobileNetV2 on tighter person crops
   from the new YOLO — this would likely push frame classification accuracy above 98%
