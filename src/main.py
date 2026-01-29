"""
Digital Witness - Main Pipeline Orchestrator

Core analysis pipeline that coordinates the processing flow:
video → pose estimation → feature extraction → classification →
POS cross-check → intent scoring → alert generation.

This module is the primary interface for running end-to-end analysis.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from .config import (
    DEFAULT_VIDEO_PATH,
    DEFAULT_POS_PATH,
    BEHAVIOR_MODEL_PATH,
    OUTPUTS_DIR,
    CASE_OUTPUT_DIR
)
from .video import VideoLoader, VideoMetadata, ClipExtractor
from .pose import PoseEstimator, FeatureExtractor, BehaviorClassifier, BehaviorEvent
from .pos import POSDataLoader, Transaction, TransactionItem
from .analysis import CrossChecker, ProductInteraction, IntentScorer, AlertGenerator, Severity


def run_pipeline(
    video_path: Optional[Path] = None,
    pos_path: Optional[Path] = None,
    product_mapping: Optional[Dict[float, str]] = None
) -> dict:
    """
    Execute the complete analysis pipeline.

    Args:
        video_path: Source video file path
        pos_path: POS transaction JSON file path
        product_mapping: Timestamp-to-SKU mapping for cross-checking

    Returns:
        Dict containing all intermediate and final results
    """
    video_path = video_path or DEFAULT_VIDEO_PATH
    pos_path = pos_path or DEFAULT_POS_PATH

    print("=" * 60)
    print("  DIGITAL WITNESS - Retail Security Analysis")
    print("=" * 60)
    print()

    results = {}

    # Step 1: Load video
    print("[1/8] Loading video...")
    try:
        video_loader = VideoLoader(video_path)
        metadata = video_loader.metadata
        print(f"  - File: {metadata.path.name}")
        print(f"  - Duration: {metadata.duration:.1f}s")
        print(f"  - Resolution: {metadata.width}x{metadata.height}")
        print(f"  - FPS: {metadata.fps}")
        results["video_metadata"] = metadata
    except FileNotFoundError:
        print(f"  ! Video file not found: {video_path}")
        print("  ! Running in demo mode with simulated data...")
        return run_demo_mode(pos_path)

    # Step 2: Run pose estimation
    print("\n[2/8] Running pose estimation...")
    pose_results = []
    with video_loader:
        with PoseEstimator() as estimator:
            frame_count = 0
            for frame_num, frame in video_loader.frames(step=2):
                result = estimator.process_frame(frame, frame_num, metadata.fps)
                pose_results.append(result)
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"  - Processed {frame_count} frames...")

    poses_detected = sum(1 for p in pose_results if p.landmarks is not None)
    print(f"  - Total frames processed: {frame_count}")
    print(f"  - Poses detected: {poses_detected}")
    results["pose_results"] = pose_results

    # Step 3: Extract features
    print("\n[3/8] Extracting features from pose sequences...")
    feature_extractor = FeatureExtractor()
    pose_features = feature_extractor.extract_from_sequence(pose_results)
    print(f"  - Feature windows extracted: {len(pose_features)}")
    results["pose_features"] = pose_features

    # Step 4: Classify behaviors
    print("\n[4/8] Classifying behaviors...")
    try:
        classifier = BehaviorClassifier()
        behavior_events = classifier.classify_sequence(pose_features)
        print(f"  - Behavior events detected: {len(behavior_events)}")
        for btype in ["normal", "shoplifting"]:
            count = sum(1 for e in behavior_events if e.behavior_type == btype)
            if count > 0:
                print(f"    - {btype}: {count}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ! Model not trained: {e}")
        print("  ! Run 'python run.py --train' first")
        behavior_events = []
    results["behavior_events"] = behavior_events

    # Step 5: Load POS data
    print("\n[5/8] Loading POS transaction data...")
    transactions = []
    try:
        pos_loader = POSDataLoader(pos_path)
        transactions = pos_loader.load()
        print(f"  - Transactions loaded: {len(transactions)}")
        total_items = sum(len(t.items) for t in transactions)
        print(f"  - Total items billed: {total_items}")
    except FileNotFoundError:
        print(f"  ! POS file not found: {pos_path}")
        print("  ! Using empty transaction list for analysis")
    results["transactions"] = transactions

    # Step 6: Cross-check
    print("\n[6/8] Cross-checking detected interactions with billing...")
    cross_checker = CrossChecker()
    detected_interactions = []
    demo_skus = ["ITEM001", "ITEM002", "ITEM003"]
    for i, event in enumerate(behavior_events):
        if event.behavior_type in ["pickup", "shoplifting"] and i < len(demo_skus):
            interaction = ProductInteraction(
                sku=demo_skus[i],
                timestamp=event.start_time,
                interaction_type="pickup",
                confidence=event.confidence
            )
            detected_interactions.append(interaction)

    discrepancy_report = cross_checker.check_discrepancies(detected_interactions, transactions)
    print(f"  - Items detected: {discrepancy_report.total_detected}")
    print(f"  - Items billed: {discrepancy_report.total_billed}")
    print(f"  - Discrepancies: {discrepancy_report.discrepancy_count}")
    results["discrepancy_report"] = discrepancy_report

    # Step 7: Calculate intent score
    print("\n[7/8] Calculating intent score...")
    intent_scorer = IntentScorer()
    intent_score = intent_scorer.calculate_score(discrepancy_report, behavior_events, metadata.duration)
    print(f"  - Intent score: {intent_score.score:.2f}")
    print(f"  - Severity: {intent_score.severity.value}")
    results["intent_score"] = intent_score

    # Step 8: Generate alert
    print("\n[8/8] Checking alert conditions...")
    alert_generator = AlertGenerator()

    # Extract forensic clips for suspicious events
    suspicious_events = [
        {"timestamp": e.start_time, "type": e.behavior_type}
        for e in behavior_events
        if e.behavior_type in ["concealment", "bypass", "shoplifting"]
    ]

    forensic_clips = []
    if suspicious_events:
        try:
            with video_loader:
                clip_extractor = ClipExtractor(video_loader)
                forensic_clips = clip_extractor.extract_clips_for_events(suspicious_events[:5])
            print(f"  - Clips extracted: {len(forensic_clips)}")
        except Exception as e:
            print(f"  - Clip extraction skipped: {e}")

    alert = alert_generator.generate_alert(intent_score, discrepancy_report, behavior_events, forensic_clips)
    if alert:
        print(f"\n  *** ALERT GENERATED ***")
        print(f"  - Alert ID: {alert.alert_id}")
        print(f"  - Severity: {alert.severity.value}")
    else:
        print("  - No alert generated (below threshold)")
    results["alert"] = alert
    results["forensic_clips"] = forensic_clips

    # Final summary
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Risk Score: {intent_score.score:.2f} ({intent_score.severity.value})")
    print(f"  Suspicious Events: {len([e for e in behavior_events if e.behavior_type != 'normal'])}")
    print(f"  Billing Discrepancies: {discrepancy_report.discrepancy_count}")
    if alert:
        print(f"\n  ALERT: {alert.alert_id}")
        print(f"  {alert.explanation[:200]}...")

    return results


def run_demo_mode(pos_path: Optional[Path] = None) -> dict:
    """Run in demo mode without actual video file."""
    print("\n" + "=" * 60)
    print("  DEMO MODE - Simulated Analysis")
    print("=" * 60)

    # Simulate video metadata
    metadata = VideoMetadata(
        path=Path("demo_video.mp4"),
        fps=30.0,
        frame_count=1800,
        duration=60.0,
        width=1920,
        height=1080,
        codec="H264"
    )

    # Simulate behavior events
    behavior_events = [
        BehaviorEvent("normal", 0.0, 10.0, 0, 300, 0.9, {"normal": 0.9, "shoplifting": 0.1}),
        BehaviorEvent("shoplifting", 10.0, 15.0, 300, 450, 0.85, {"normal": 0.15, "shoplifting": 0.85}),
        BehaviorEvent("normal", 15.0, 50.0, 450, 1500, 0.92, {"normal": 0.92, "shoplifting": 0.08}),
        BehaviorEvent("shoplifting", 50.0, 60.0, 1500, 1800, 0.78, {"normal": 0.22, "shoplifting": 0.78}),
    ]

    # Load POS data
    print("\n[1/4] Loading POS data...")
    transactions = []
    try:
        pos_loader = POSDataLoader(pos_path or DEFAULT_POS_PATH)
        transactions = pos_loader.load()
        print(f"  - Transactions: {len(transactions)}")
    except FileNotFoundError:
        print("  - No POS data found, using empty list")

    # Cross-check
    print("\n[2/4] Cross-checking...")
    cross_checker = CrossChecker()
    detected_interactions = [
        ProductInteraction("ITEM001", 10.0, "pickup", 0.85),
        ProductInteraction("ITEM002", 50.0, "pickup", 0.78),
    ]
    discrepancy_report = cross_checker.check_discrepancies(detected_interactions, transactions)
    print(f"  - Discrepancies: {discrepancy_report.discrepancy_count}")

    # Intent score
    print("\n[3/4] Calculating intent score...")
    intent_scorer = IntentScorer()
    intent_score = intent_scorer.calculate_score(discrepancy_report, behavior_events, 60.0)
    print(f"  - Score: {intent_score.score:.2f} ({intent_score.severity.value})")

    # Alert
    print("\n[4/4] Generating alert...")
    alert_generator = AlertGenerator()
    alert = alert_generator.generate_alert(intent_score, discrepancy_report, behavior_events, [])

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Risk Score: {intent_score.score:.2f} ({intent_score.severity.value})")

    if alert:
        print(f"\n  ALERT GENERATED: {alert.alert_id}")

    return {
        "video_metadata": metadata,
        "behavior_events": behavior_events,
        "transactions": transactions,
        "discrepancy_report": discrepancy_report,
        "intent_score": intent_score,
        "alert": alert
    }


def main():
    """Main entry point."""
    video_path = None
    pos_path = None

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        pos_path = Path(sys.argv[2])

    run_pipeline(video_path, pos_path)


if __name__ == "__main__":
    main()
