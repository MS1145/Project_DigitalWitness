# Implementation Research: CNN-LSTM Architectures for Shoplifting Detection

This document summarizes research on deep learning approaches for video-based shoplifting and anomaly detection, compared against the Digital Witness architecture.

---

## Table of Contents

1. [Current Digital Witness Architecture](#current-digital-witness-architecture)
2. [Alternative Architectures Reviewed](#alternative-architectures-reviewed)
   - [Architecture A: Custom CNN + LSTM](#architecture-a-custom-cnn--lstm)
   - [Architecture B: 2D CNN + GRU](#architecture-b-2d-cnn--gru)
3. [Research Papers](#research-papers)
   - [3D CNN: Pre-Crime Behavior Detection](#1-3d-cnn-pre-crime-behavior-detection)
   - [π-VAD: Poly-modal Induced Video Anomaly Detection](#2-π-vad-poly-modal-induced-video-anomaly-detection)
   - [GS-MoE: Gaussian Splatting-guided Mixture of Experts](#3-gs-moe-gaussian-splatting-guided-mixture-of-experts)
4. [Comparative Analysis](#comparative-analysis)
5. [Key Takeaways for Digital Witness](#key-takeaways-for-digital-witness)
6. [Potential Improvements](#potential-improvements)

---

## Current Digital Witness Architecture

| Component | Technology | Specification |
|-----------|------------|---------------|
| **Object Detection** | YOLOv8 | Person/product detection & tracking |
| **Feature Extraction** | ResNet18 (pretrained) | 512-dimensional features |
| **Temporal Classification** | Bidirectional LSTM + Attention | 256 hidden dim, 2 layers |
| **Output Classes** | Binary | Normal, Shoplifting |

### Architecture Diagram

```
Video Frame
     ↓
┌─────────────┐
│   YOLOv8    │ → Object Detection (persons, products)
└─────────────┘
     ↓
┌─────────────┐
│  ByteTrack  │ → Multi-Object Tracking
└─────────────┘
     ↓
┌─────────────┐
│  ResNet18   │ → 512-dim Spatial Features
│ (pretrained)│
└─────────────┘
     ↓
┌─────────────┐
│ Bi-LSTM +   │ → Temporal Pattern Learning
│  Attention  │    (256 hidden, 2 layers)
└─────────────┘
     ↓
┌─────────────┐
│  Classifier │ → Binary: Normal / Shoplifting
└─────────────┘
```

### Current Model Performance

- **Validation Accuracy:** 94.8%
- **Training Samples:** 3,028 sequences
- **Dataset:** 90 normal videos, 92 shoplifting videos

---

## Alternative Architectures Reviewed

### Architecture A: Custom CNN + LSTM

**Source:** External project reference

| Component | Specification |
|-----------|---------------|
| **CNN Encoder** | Conv2D + ReLU + MaxPooling → Flatten → FC(256) |
| **LSTM** | Hidden size = 128, batch-first mode |
| **Output** | Binary classification (shoplifter / non-shoplifter) |

#### Comparison with Digital Witness

| Aspect | Architecture A | Digital Witness |
|--------|---------------|-----------------|
| **CNN** | Custom shallow (256 features) | ResNet18 pretrained (512 features) |
| **LSTM** | Unidirectional, 128 hidden | Bidirectional + Attention, 256 hidden |
| **Transfer Learning** | None | ImageNet weights |
| **Attention** | No | Yes |

#### Assessment

- **Pros:** Simpler, faster training, fewer parameters
- **Cons:** Requires more training data, less robust feature extraction
- **Verdict:** Digital Witness architecture is more sophisticated and likely performs better with limited data due to transfer learning

---

### Architecture B: 2D CNN + GRU

**Source:** External project reference

| Component | Specification |
|-----------|---------------|
| **Feature Extraction** | 2D CNN |
| **Temporal Modeling** | GRU (Gated Recurrent Unit) |
| **Output** | Binary classification |

#### GRU vs LSTM Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (input, forget, output) | 2 (reset, update) |
| **Parameters** | More (~33% more) | Fewer |
| **Training Speed** | Slower | Faster |
| **Long Sequences** | Better for very long dependencies | Good for moderate sequences |
| **Performance** | Often similar | Often similar |

#### Assessment

- **Pros:** Faster training/inference, less overfitting risk, simpler to debug
- **Cons:** May struggle with very long video sequences
- **Verdict:** GRU is a valid alternative if experiencing overfitting or need faster inference. Not necessarily better than LSTM.

---

## Research Papers

### 1. 3D CNN: Pre-Crime Behavior Detection

**Paper:** "Suspicious Behavior Detection on Shoplifting Cases"
**Authors:** Martínez-Mascorro et al.

#### Core Objective

Focus on **pre-crime prevention** - detecting suspicious behavior (looking around, nervous gestures) BEFORE the actual theft occurs.

#### Technical Architecture

```
Video Segments (grayscale)
        ↓
┌───────────────────────┐
│   4x Conv3D Layers    │ → Spatiotemporal feature extraction
│   (32 + 64 filters)   │    Kernel: 3×3×3
└───────────────────────┘
        ↓
┌───────────────────────┐
│   2x MaxPooling3D     │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Dense(512) → Dense(2)│ → Softmax binary classification
└───────────────────────┘
```

#### Key Innovation: Pre-Crime Behavior (PCB) Analysis

Novel data labeling methodology:

| Segment | Definition |
|---------|------------|
| **SCM** (Strict Crime Moment) | The exact segment where theft occurs |
| **CCM** (Comprehensive Crime Moment) | When intentions become obvious (checking for cameras) |
| **PCB** (Pre-Crime Behavior) | From subject's first appearance until CCM - contains behavioral cues WITHOUT actual theft |

#### Results

- **Accuracy:** 75% for pre-crime detection
- **Optimal Configuration:** 80×60 pixel resolution, 10-frame depth

#### Relevance to Digital Witness

- PCB labeling concept could improve training data annotation
- 3D CNN captures motion in single operation (vs separate CNN + LSTM)
- Trade-off: Less interpretable than attention-based approach

---

### 2. π-VAD: Poly-modal Induced Video Anomaly Detection

**Paper:** "PI-VAD: Poly-modal Induced Video Anomaly Detection"
**Authors:** Majhi et al.

#### Core Objective

Address limitation of RGB-only systems that struggle to distinguish complex anomalies (shoplifting, abuse) from visually similar normal events.

#### Technical Architecture

**Five Modalities Used (Training Only):**

| Modality | Purpose |
|----------|---------|
| **Pose** | Fine-grained motion analysis |
| **Depth** | 3D geometry understanding |
| **Panoptic Masks** | Object segmentation |
| **Optical Flow** | Global motion patterns |
| **Text/Language** | Semantic understanding via VLMs |

#### Teacher-Student Framework

```
Training Phase:
┌─────────────────────────────────────────────────┐
│                   TEACHER                        │
│  (Frozen, pre-trained on multi-modal data)      │
└─────────────────────────────────────────────────┘
                      ↓ Knowledge Distillation
┌─────────────────────────────────────────────────┐
│                   STUDENT                        │
│  ┌─────────────────────────────────────────┐   │
│  │  Pseudo Modality Generation (PMG)       │   │
│  │  - Generates pseudo embeddings for      │   │
│  │    5 modalities from RGB features       │   │
│  └─────────────────────────────────────────┘   │
│                      ↓                          │
│  ┌─────────────────────────────────────────┐   │
│  │  Cross Modal Induction (CMI)            │   │
│  │  - Bi-directional InfoNCE loss          │   │
│  │  - Aligns RGB with multi-modal cues     │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘

Inference Phase:
┌─────────────────────────────────────────────────┐
│  RGB Input Only → Student Model → Prediction    │
│  (No heavy modality-specific backbones needed)  │
└─────────────────────────────────────────────────┘
```

#### Results

- **UCF-Crime AUC:** 90.33% (State-of-the-Art)
- **Abnormal AUC (AUCA):** 77.77%
- **Inference Speed:** 30 FPS (real-time viable)

#### Relevance to Digital Witness

- Digital Witness already has MediaPipe pose - could leverage it more
- Teacher-Student approach enables rich training without inference overhead
- Multi-modal fusion significantly improves anomaly discrimination

---

### 3. GS-MoE: Gaussian Splatting-guided Mixture of Experts

**Paper:** "GS-MoE for Weakly-Supervised Video Anomaly Detection"
**Authors:** D'Amicantonio et al.

#### Core Objective

Address two failures in Weakly-Supervised VAD:

1. **Coarse Supervision:** MIL only focuses on "top-k" abnormal snippets, ignoring temporal duration
2. **Class Confusion:** Single shared model fails to capture class-specific anomaly cues

#### Technical Architecture

##### Temporal Gaussian Splatting (TGS)

```
Traditional MIL:
Frame scores: [0.2, 0.3, 0.9, 0.8, 0.3, 0.2]
                        ↑    ↑
                    Only these "top-k" frames used
                    (ignores temporal context)

Gaussian Splatting:
Frame scores: [0.2, 0.3, 0.9, 0.8, 0.3, 0.2]
                        ↓
Gaussian kernel around peaks:
              [0.3, 0.6, 0.9, 0.8, 0.5, 0.3]
                   ↑              ↑
              Smooth continuous pseudo-labels
              (captures ENTIRE anomaly duration)
```

##### Mixture of Experts (MoE)

```
┌─────────────────────────────────────────────────┐
│           Feature Extraction                     │
│           (I3D + UR-DMU)                        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Expert Router (Gating Network)          │
└─────────────────────────────────────────────────┘
         ↓           ↓           ↓           ↓
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Expert  │   │ Expert  │   │ Expert  │   │ Expert  │
│ (Arson) │   │(Assault)│   │(Shoplift)│  │ (etc.) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
    ↓             ↓             ↓             ↓
    └─────────────┴─────────────┴─────────────┘
                      ↓
              Combined Prediction
```

#### Results

- Improved detection of temporally extended anomalies
- Better class-specific discrimination
- Reduced false positives from class confusion

#### Relevance to Digital Witness

- Gaussian pseudo-labels could improve training for behaviors spanning multiple frames
- MoE concept interesting for distinguishing pickup vs concealment vs bypass
- Would require significant architectural changes

---

## Comparative Analysis

### Architecture Comparison Table

| Feature | Digital Witness | 3D CNN | π-VAD | GS-MoE |
|---------|----------------|--------|-------|--------|
| **Spatial Features** | ResNet18 | 3D Conv | Multi-modal | I3D |
| **Temporal Modeling** | Bi-LSTM + Attention | 3D Conv | Teacher-Student | MoE + Gaussian |
| **Pre-training** | ImageNet | None | Multi-modal | Video datasets |
| **Explainability** | Attention weights | Limited | Limited | Expert routing |
| **Real-time** | Yes | Yes | Yes (30 FPS) | Depends |
| **Training Data Needed** | Moderate | High | High (multi-modal) | Moderate |

### Complexity vs Performance Trade-off

```
Performance
    ↑
    │                              ┌─────┐
    │                         ┌────│π-VAD│
    │                    ┌────┘    └─────┘
    │               ┌────┘    ┌──────────┐
    │          ┌────┘    ┌────│Digital   │
    │     ┌────┘         │    │Witness   │
    │┌────┘              │    └──────────┘
    ││    ┌──────┐       │         ┌──────┐
    ││    │Custom│       │         │GS-MoE│
    ││    │CNN+  │       │         └──────┘
    ││    │LSTM  │       │
    │└────┴──────┘    ┌──┴──┐
    │                 │3D   │
    │                 │CNN  │
    │                 └─────┘
    └─────────────────────────────────────→ Complexity
```

---

## Key Takeaways for Digital Witness

### What's Working Well

1. **ResNet18 backbone** - Transfer learning provides robust features without massive training data
2. **Bidirectional LSTM** - Captures both past and future context
3. **Attention mechanism** - Provides explainability (which frames mattered)
4. **94.8% validation accuracy** - Strong baseline performance

### Insights from Research

| Research | Key Insight | Applicability |
|----------|-------------|---------------|
| **3D CNN** | Pre-crime behavior labeling | Could improve training data annotation |
| **π-VAD** | Multi-modal fusion at training | Already have pose via MediaPipe - underutilized |
| **GS-MoE** | Class-specific experts | Could help distinguish pickup/concealment/bypass |

### Current Issues Identified

1. **Aggregation Bias:** 2x weight for suspicious classes + 30% threshold causes false positives
2. **Domain Shift:** Model may not generalize well to videos different from training set
3. **Binary vs Multi-class:** Training uses 2 classes but config defines 4 classes

---

## Potential Improvements

### Short-term (No Architecture Changes)

| Improvement | Implementation | Effort |
|-------------|----------------|--------|
| Reduce aggregation bias | Change `suspicious_weight` from 2.0 to 1.0 | Low |
| Raise alert threshold | Change threshold from 0.3 to 0.5 | Low |
| Add more diverse training data | Collect videos from different environments | Medium |
| Data augmentation | Horizontal flip, brightness/contrast variations | Medium |

### Medium-term (Minor Architecture Changes)

| Improvement | Implementation | Effort |
|-------------|----------------|--------|
| Leverage MediaPipe pose | Add pose features to CNN input | Medium |
| Add confidence calibration | Temperature scaling on outputs | Medium |
| Implement GRU variant | Compare LSTM vs GRU performance | Medium |

### Long-term (Major Architecture Changes)

| Improvement | Implementation | Effort |
|-------------|----------------|--------|
| 3D CNN branch | Add parallel 3D conv for motion features | High |
| Teacher-Student training | Multi-modal training, RGB-only inference | High |
| Mixture of Experts | Separate experts for different behavior types | High |
| Pre-crime behavior detection | Relabel training data with PCB methodology | High |

---

## References

1. Martínez-Mascorro et al. - "Suspicious Behavior Detection on Shoplifting Cases for Crime Prevention by Using 3D Convolutional Neural Networks"
2. Majhi et al. - "π-VAD: Poly-modal Induced Video Anomaly Detection"
3. D'Amicantonio et al. - "GS-MoE: Gaussian Splatting-guided Mixture of Experts for Video Anomaly Detection"

---

*Document created: 2026-02-04*
*Project: Digital Witness - Retail Security Assistant*
