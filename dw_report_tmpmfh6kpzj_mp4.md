# Digital Witness — Analysis Report
**Generated:** 2026-03-27 10:42:48

---
## Video Metadata
| Field | Value |
|---|---|
| Filename | tmpqv4d60xu.mp4 |
| Duration | 22.0s |
| FPS | 30.0 |
| Resolution | 768×432 |
| Frame Count | 658 |

---
## Detection Result
| Field | Value |
|---|---|
| Classification | **SHOPLIFTING** |
| Confidence | 82.6% |
| Is Shoplifting | True |
| Risk Level | MEDIUM |
| Risk Score | 0.619 |
| YOLO Override Suspected | True |

---
## Score Components
| Component | Score |
|---|---|
| Concealment | 0.000 |
| Bypass | 0.000 |
| Duration | 0.000 |

> **Note:** All components are 0 — detection was YOLO-driven, not LSTM temporal.

**Explanation:** Score: 0.62 (MEDIUM). Concealment: 0.00, Bypass: 0.00, Duration: 0.00

---
## Detection Statistics
| Metric | Value |
|---|---|
| People Interacted with Products | 2 |
| Products Detected | 3 |
| Suspicious Interactions | 0 |
| Frames Processed | 164 |

---
## Behavior Events (8 windows)
| # | Behavior | Start | End | Confidence |
|---|---|---|---|---|
| 1 | Normal | 0.0s | 6.0s | 97.1% |
| 2 | Normal | 2.0s | 8.0s | 97.1% |
| 3 | Normal | 4.0s | 10.0s | 97.1% |
| 4 | Normal | 6.0s | 12.0s | 97.1% |
| 5 | Normal | 8.0s | 14.0s | 97.2% |
| 6 | Normal | 10.0s | 16.0s | 97.2% |
| 7 | Normal | 12.0s | 18.0s | 97.2% |
| 8 | Normal | 14.0s | 20.0s | 97.2% |

> **Note:** All LSTM windows labelled 'normal' — timeline mismatch with SHOPLIFTING banner is expected (YOLO override).

---
## Forensic Clips (4 extracted)
| # | Behavior | Start | End | Confidence |
|---|---|---|---|---|
| 1 | NORMAL | 9.0s | 17.0s | 97% |
| 2 | NORMAL | 11.0s | 19.0s | 97% |
| 3 | NORMAL | 13.0s | 21.0s | 97% |
| 4 | NORMAL | 7.0s | 15.0s | 97% |

---
## Quality & Fairness
| Metric | Value |
|---|---|
| Reliability Score | 85.0% |
| Detection Rate | 100.0% |
| Analysis Usable | True |
| Fairness Score | 85.0% |
| Requires Review | True |

---
## Raw JSON (for debugging)
```json
{
  "success": true,
  "video_metadata": {
    "filename": "tmpqv4d60xu.mp4",
    "duration": 21.955266666666667,
    "fps": 29.97002997002997,
    "width": 768,
    "height": 432,
    "frame_count": 658
  },
  "lstm_detection": {
    "classification": "shoplifting",
    "confidence": 0.825818657875061,
    "is_shoplifting": true
  },
  "detections": {
    "persons_tracked": 2,
    "products_detected": 3,
    "interactions": 0,
    "frames_processed": 164
  },
  "product_pickups": {
    "Looking around": 3,
    "Picking-Holding": 4,
    "shoplifting": 3
  },
  "behavior_events": [
    {
      "behavior_type": "normal",
      "start_time": 0.0,
      "end_time": 6.006,
      "confidence": 0.9712653160095215,
      "probabilities": {
        "normal": 0.9712653160095215,
        "shoplifting": 0.02873467281460762
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 2.0020000000000002,
      "end_time": 8.008000000000001,
      "confidence": 0.9709314703941345,
      "probabilities": {
        "normal": 0.9709314703941345,
        "shoplifting": 0.02906855009496212
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 4.0040000000000004,
      "end_time": 10.01,
      "confidence": 0.9713402390480042,
      "probabilities": {
        "normal": 0.9713402390480042,
        "shoplifting": 0.028659742325544357
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 6.006,
      "end_time": 12.012,
      "confidence": 0.9712927937507629,
      "probabilities": {
        "normal": 0.9712927937507629,
        "shoplifting": 0.028707196936011314
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 8.008000000000001,
      "end_time": 14.014000000000001,
      "confidence": 0.9718228578567505,
      "probabilities": {
        "normal": 0.9718228578567505,
        "shoplifting": 0.028177065774798393
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 10.01,
      "end_time": 16.016000000000002,
      "confidence": 0.9721141457557678,
      "probabilities": {
        "normal": 0.9721141457557678,
        "shoplifting": 0.027885926887392998
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 12.012,
      "end_time": 18.018,
      "confidence": 0.9721131324768066,
      "probabilities": {
        "normal": 0.9721131324768066,
        "shoplifting": 0.027886882424354553
      }
    },
    {
      "behavior_type": "normal",
      "start_time": 14.014000000000001,
      "end_time": 20.02,
      "confidence": 0.9720240235328674,
      "probabilities": {
        "normal": 0.9720240235328674,
        "shoplifting": 0.027976034209132195
      }
    }
  ],
  "intent_score": {
    "score": 0.6193639934062958,
    "severity": "MEDIUM",
    "components": {
      "concealment": 0.0,
      "bypass": 0.0,
      "duration": 0.0
    },
    "explanation": "Score: 0.62 (MEDIUM). Concealment: 0.00, Bypass: 0.00, Duration: 0.00"
  },
  "quality_analysis": {
    "reliability_score": 0.85,
    "detection_rate": 1.0,
    "usable": true
  },
  "bias_report": {
    "overall_fairness_score": 0.85,
    "analysis_reliable": true,
    "requires_review": true,
    "flags": []
  },
  "alert": {
    "alert_id": "ALERT-20260327104215-0001",
    "level": "MEDIUM",
    "message": "6 suspicious event(s) detected. Adjusted risk score: 0.62 (MEDIUM). Human review required before any action."
  },
  "model_trained": true,
  "annotated_video_path": "C:\\Users\\MSI\\AppData\\Local\\Temp\\dw_annotated_e312ee7e.mp4",
  "suspicious_frames": [
    {
      "timestamp": 13.013000000000002,
      "frame_start": 300,
      "frame_end": 480,
      "clip_start": 9.01,
      "clip_end": 17.016000000000002,
      "behavior": "normal",
      "confidence": 0.9721141457557678
    },
    {
      "timestamp": 15.015,
      "frame_start": 360,
      "frame_end": 540,
      "clip_start": 11.012,
      "clip_end": 19.018,
      "behavior": "normal",
      "confidence": 0.9721131324768066
    },
    {
      "timestamp": 17.017,
      "frame_start": 420,
      "frame_end": 600,
      "clip_start": 13.014000000000001,
      "clip_end": 21.02,
      "behavior": "normal",
      "confidence": 0.9720240235328674
    },
    {
      "timestamp": 11.011000000000001,
      "frame_start": 240,
      "frame_end": 420,
      "clip_start": 7.008000000000001,
      "clip_end": 15.014000000000001,
      "behavior": "normal",
      "confidence": 0.9718228578567505
    }
  ]
}
```