# Implementation Plan: Real-Time Decision Support System

## Overview

This document outlines the implementation plan for transforming Digital Witness from a pre-recorded video analysis MVP to a real-time decision-support system.

---

## Phase 2.1: Real-Time Video Infrastructure

### Task 2.1.1: Live Camera Feed Handler

**Files to create:**

- `src/video/live_stream.py`

**Implementation:**

```python
# Core components needed:
- RTSPStreamHandler: Connect to IP cameras via RTSP
- FrameBuffer: Ring buffer for recent frames (configurable size)
- FrameGrabber: Async frame acquisition thread
- StreamHealthMonitor: Connection status and reconnection logic
```

**Dependencies:** `opencv-python`, `threading`, `queue`

**Acceptance Criteria:**

- [ ] Connect to RTSP camera stream
- [ ] Maintain 30-frame ring buffer
- [ ] Handle disconnection and auto-reconnect
- [ ] Support multiple camera feeds (future)

---

### Task 2.1.2: Real-Time Pose Processing Pipeline

**Files to modify:**

- `src/pose/estimator.py` (add streaming mode)
- `src/pose/feature_extractor.py` (add incremental extraction)

**Files to create:**

- `src/pipeline/realtime_processor.py`

**Implementation:**

```python
# Core components:
- RealtimePoseProcessor: Process frames as they arrive
- SlidingWindowManager: Maintain feature windows in real-time
- PoseTracker: Track multiple people across frames
```

**Acceptance Criteria:**

- [ ] Process frames at ≥15 FPS
- [ ] Extract features incrementally (no batch processing)
- [ ] Track pose IDs across frames

- [ ] Handle frame drops gracefully

---

## Phase 2.2: Enhanced Behaviour Detection

### Task 2.2.1: Gesture Classifier Training Data

**Files to create:**

- `data/gestures/` directory structure
- `src/pose/gesture_labeler.py` (annotation tool)

**Gestures to capture:**

1. Product pickup (hand reaching to shelf, grasping motion)
2. Concealment (hand-to-pocket, hand-to-bag)
3. Product put-back (returning item to shelf)
4. Normal walking/browsing

**Data requirements:**

- Minimum 100 samples per gesture class
- Varied lighting, angles, body types
- Include edge cases (children, elderly)

---

### Task 2.2.2: Gesture Classification Model

**Files to create:**

- `src/pose/gesture_classifier.py`
- `src/pose/train_gesture_classifier.py`

**Model architecture:**

```
Input: Pose sequence (30 frames × 33 landmarks × 3 coords)
       ↓
Temporal CNN or LSTM layer
       ↓
Dense layers with dropout
       ↓
Output: Gesture class probabilities
```

**Acceptance Criteria:**

- [ ] ≥80% accuracy on gesture classification
- [ ] <100ms inference time per sequence
- [ ] Confidence scores for each prediction

---

### Task 2.2.3: Behaviour State Machine

**Files to create:**

- `src/analysis/behaviour_tracker.py`

**State machine design:**

```
                    ┌─────────────┐
                    │   IDLE      │
                    └──────┬──────┘
                           │ enters store zone
                           ▼
                    ┌─────────────┐
          ┌────────│  BROWSING   │────────┐
          │        └──────┬──────┘        │
          │               │ pickup        │ exit without
          │               ▼               │ interaction
          │        ┌─────────────┐        │
          │        │  HOLDING    │        │
          │        └──────┬──────┘        │
          │         │     │     │         │
          │    put  │     │     │ conceal │
          │    back │     │     │         │
          │         ▼     │     ▼         │
          │   ┌─────────┐ │ ┌─────────┐   │
          │   │RETURNED │ │ │CONCEALED│   │
          │   └────┬────┘ │ └────┬────┘   │
          │        │      │      │        │
          │        └──────┼──────┘        │
          │               │               │
          │               ▼               │
          │        ┌─────────────┐        │
          └───────▶│  CHECKOUT   │◀───────┘
                   └──────┬──────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ BILLED   │ │ PARTIAL  │ │ UNBILLED │
        └──────────┘ └──────────┘ └──────────┘
```

**Acceptance Criteria:**

- [ ] Track customer state throughout journey
- [ ] Maintain interaction history per customer
- [ ] Emit state change events for logging

---

## Phase 2.3: Live POS Integration

### Task 2.3.1: POS Event Receiver

**Files to create:**

- `src/pos/event_receiver.py`
- `src/pos/event_queue.py`

**Supported integration methods:**

1. **Webhook**: POST endpoint for POS system callbacks
2. **Polling**: REST API polling for systems without webhooks
3. **File watch**: Monitor CSV/JSON export directory

**Event schema:**

```json
{
  "event_type": "item_scanned|transaction_complete|void",
  "timestamp": "ISO8601",
  "terminal_id": "POS001",
  "transaction_id": "TXN123",
  "item": {
    "sku": "ITEM001",
    "name": "Product Name",
    "price": 2.99,
    "quantity": 1
  }
}
```

**Acceptance Criteria:**

- [ ] Receive POS events within 500ms
- [ ] Queue events for processing
- [ ] Handle duplicate events (idempotency)
- [ ] Support transaction grouping

---

### Task 2.3.2: Behaviour-POS Fusion Engine

**Files to create:**

- `src/analysis/fusion_engine.py`

**Fusion logic:**

```python
# For each tracked customer:
1. Get list of detected product interactions
2. Get list of billed items from POS
3. Correlate by:
   - Temporal proximity (interaction time vs scan time)
   - Zone proximity (interaction zone vs checkout zone)
   - Product matching (if SKU detection available)
4. Identify discrepancies:
   - Interacted but not billed
   - Billed but not detected (could be edge case)
```

**Acceptance Criteria:**

- [ ] Correlate interactions with POS events
- [ ] Generate discrepancy reports
- [ ] Handle timing mismatches (configurable window)

---

## Phase 2.4: Alert & Notification System

### Task 2.4.1: Risk Scoring Engine (Enhanced)

**Files to modify:**

- `src/analysis/intent_scorer.py`

**New scoring factors:**

```python
WEIGHT_CONCEALMENT = 0.30      # Concealment gesture detected
WEIGHT_DISCREPANCY = 0.25      # Items interacted but not billed
WEIGHT_EXIT_BEHAVIOUR = 0.20   # Moving to exit without checkout
WEIGHT_DWELL_TIME = 0.10       # Unusual time in zones
WEIGHT_RETURN_RATE = 0.10      # Items put back (reduces risk)
WEIGHT_EDGE_CASE = -0.15       # Reduction for detected edge cases
```

**Acceptance Criteria:**

- [ ] Real-time score updates as events occur
- [ ] Score history for trend analysis
- [ ] Configurable weights via config file

---

### Task 2.4.2: Alert Manager

**Files to create:**

- `src/alerts/alert_manager.py`
- `src/alerts/notification_service.py`

**Alert flow:**

```
Risk Score Update
       │
       ▼
  Threshold Check ──────────────────────┐
       │                                │
       │ exceeds threshold              │ below threshold
       ▼                                ▼
  Generate Alert                   Log only
       │
       ▼
  Create Forensic Package
       │
       ▼
  Send Notification
       │
       ├── Dashboard (WebSocket)
       ├── Email (SMTP)
       ├── SMS (Twilio)
       └── Push (Firebase)
```

**Acceptance Criteria:**

- [ ] Configurable thresholds per risk level
- [ ] Multiple notification channels
- [ ] Alert deduplication (don't spam for same customer)
- [ ] Alert acknowledgment tracking

---

### Task 2.4.3: Forensic Package Generator

**Files to create:**

- `src/forensics/package_builder.py`
- `src/forensics/clip_annotator.py`

**Package contents:**

```
forensic_package_<timestamp>/
├── summary.json           # Alert details, risk score breakdown
├── timeline.json          # Chronological event list
├── clips/
│   ├── pickup_001.mp4     # Product pickup moment
│   ├── concealment_001.mp4
│   ├── checkout_area.mp4
│   └── exit.mp4
├── poses/
│   ├── keyframes.json     # Pose data at key moments
│   └── trajectory.png     # Movement path visualization
└── explanation.txt        # Human-readable summary
```

**Acceptance Criteria:**

- [ ] Generate package within 5 seconds of alert
- [ ] Include all relevant video clips
- [ ] Annotate clips with timestamps and labels
- [ ] Generate human-readable explanation

---

## Phase 2.5: Edge Case Detection

### Task 2.5.1: Edge Case Classifier

**Files to create:**

- `src/analysis/edge_case_detector.py`

**Detection methods:**

| Edge Case         | Detection Method                                      |
| ----------------- | ----------------------------------------------------- |
| Child             | Pose height estimation, movement patterns             |
| Elderly           | Slower movement velocity, extended dwell times        |
| Barcode failure   | Repeated scan attempts at same terminal               |
| Personal items    | Track items from entry (brought in vs picked up)      |
| Put-back          | Reverse of pickup gesture, item returns to shelf zone |
| Confused customer | Erratic movement, multiple direction changes          |

**Acceptance Criteria:**

- [ ] Detect and flag edge cases
- [ ] Reduce risk score appropriately
- [ ] Include edge case info in forensic package

---

### Task 2.5.2: Zone-Based Tracking

**Files to create:**

- `src/tracking/zone_manager.py`

**Store zones:**

```
┌────────────────────────────────────────┐
│                ENTRANCE                 │
├────────────────────────────────────────┤
│                                        │
│   ┌──────┐   ┌──────┐   ┌──────┐      │
│   │SHELF │   │SHELF │   │SHELF │      │
│   │ZONE 1│   │ZONE 2│   │ZONE 3│      │
│   └──────┘   └──────┘   └──────┘      │
│                                        │
│              AISLE ZONE                │
│                                        │
├────────────────────────────────────────┤
│     CHECKOUT ZONE      │   EXIT ZONE   │
│   ┌─────┐   ┌─────┐   │               │
│   │POS 1│   │POS 2│   │               │
│   └─────┘   └─────┘   │               │
└────────────────────────┴───────────────┘
```

**Acceptance Criteria:**

- [ ] Define zones via configuration
- [ ] Track customer zone transitions
- [ ] Trigger events on zone entry/exit

---

## Phase 2.6: Real-Time Dashboard

### Task 2.6.1: Live Monitoring Dashboard

**Files to modify:**

- `app.py` (add real-time tab)

**Files to create:**

- `src/dashboard/websocket_server.py`
- `src/dashboard/live_feed.py`

**Dashboard features:**

1. Live video feed with pose overlay
2. Real-time customer tracking panel
3. Active alerts panel
4. Risk score gauges
5. Recent events timeline
6. Quick actions (acknowledge, dismiss, escalate)

**Acceptance Criteria:**

- [ ] <1 second latency for video feed
- [ ] Real-time alert updates via WebSocket
- [ ] Interactive alert management
- [ ] Multi-camera view support

---

## Implementation Timeline

```
Week 1-2:   Phase 2.1 (Real-Time Video Infrastructure)
Week 3-4:   Phase 2.2 (Enhanced Behaviour Detection)
Week 5-6:   Phase 2.3 (Live POS Integration)
Week 7-8:   Phase 2.4 (Alert & Notification System)
Week 9-10:  Phase 2.5 (Edge Case Detection)
Week 11-12: Phase 2.6 (Real-Time Dashboard)
Week 13-14: Integration Testing & Bug Fixes
Week 15-16: Documentation & Final Presentation
```

---

## File Structure (After Implementation)

```
src/
├── config.py                    # Configuration (updated)
├── main.py                      # CLI entry point
├── video/
│   ├── loader.py               # Existing
│   ├── clip_extractor.py       # Existing
│   └── live_stream.py          # NEW: RTSP handler
├── pose/
│   ├── estimator.py            # Updated for streaming
│   ├── feature_extractor.py    # Updated for incremental
│   ├── behavior_classifier.py  # Existing
│   ├── gesture_classifier.py   # NEW: Fine-grained gestures
│   └── train_gesture_classifier.py  # NEW
├── pos/
│   ├── data_loader.py          # Existing
│   ├── mock_generator.py       # Existing
│   ├── event_receiver.py       # NEW: Webhook/API
│   └── event_queue.py          # NEW: Event buffer
├── analysis/
│   ├── cross_checker.py        # Existing
│   ├── intent_scorer.py        # Updated with new weights
│   ├── alert_generator.py      # Existing
│   ├── fusion_engine.py        # NEW: Behaviour-POS fusion
│   ├── behaviour_tracker.py    # NEW: State machine
│   └── edge_case_detector.py   # NEW
├── tracking/
│   └── zone_manager.py         # NEW: Zone definitions
├── alerts/
│   ├── alert_manager.py        # NEW
│   └── notification_service.py # NEW
├── forensics/
│   ├── package_builder.py      # NEW
│   └── clip_annotator.py       # NEW
├── pipeline/
│   └── realtime_processor.py   # NEW: Main pipeline
├── dashboard/
│   ├── websocket_server.py     # NEW
│   └── live_feed.py            # NEW
└── output/
    └── case_builder.py         # Existing
```

---

## Dependencies to Add

```
# requirements.txt additions
redis                  # Event queue (optional, can use in-memory)
websockets            # Real-time dashboard updates
aiohttp               # Async HTTP for webhooks
twilio                # SMS notifications (optional)
firebase-admin        # Push notifications (optional)
```

---

## Configuration Updates

```python
# src/config.py additions

# Real-time settings
STREAM_BUFFER_SIZE = 30          # Frames to buffer
STREAM_FPS_TARGET = 15           # Target processing FPS
POSE_TRACKING_ENABLED = True     # Multi-person tracking

# POS Integration
POS_WEBHOOK_PORT = 5001
POS_POLL_INTERVAL = 1.0          # seconds
POS_CORRELATION_WINDOW = 30      # seconds

# Alert thresholds
ALERT_THRESHOLD_LOW = 0.3
ALERT_THRESHOLD_MEDIUM = 0.5
ALERT_THRESHOLD_HIGH = 0.7
ALERT_THRESHOLD_CRITICAL = 0.85

# Notification settings
NOTIFICATION_CHANNELS = ["dashboard", "email"]
NOTIFICATION_COOLDOWN = 60       # seconds between alerts for same customer

# Zone configuration
ZONES_CONFIG_PATH = "config/zones.json"

# Forensic settings
FORENSIC_CLIP_DURATION = 10      # seconds per clip
FORENSIC_RETENTION_DAYS = 30
```

---

## Risk Mitigation

| Risk                         | Mitigation                                                       |
| ---------------------------- | ---------------------------------------------------------------- |
| Real-time performance issues | Profile early, use GPU acceleration, reduce resolution if needed |
| POS integration complexity   | Start with file-based mock, add webhook later                    |
| False positive alerts        | Conservative thresholds, edge case detection                     |
| Privacy concerns             | No facial recognition, configurable retention, access controls   |
| Network failures             | Local buffering, graceful degradation                            |

---

## Testing Strategy

1. **Unit Tests**: Each new module with pytest
2. **Integration Tests**: Pipeline end-to-end with mock camera
3. **Performance Tests**: FPS benchmarks, latency measurements
4. **Edge Case Tests**: Dedicated test videos for each edge case
5. **User Acceptance**: Demo with stakeholders, iterate on feedback
