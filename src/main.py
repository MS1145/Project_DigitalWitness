"""
Digital Witness - Main Orchestration Script

Bias-Aware, Explainable Retail Security Assistant
MVP Prototype for IPD

This script runs the complete analysis pipeline:
1. Load video and extract frames
2. Run pose estimation on frames
3. Extract features from pose sequences
4. Classify behaviors using ML model
5. Load POS transaction data
6. Cross-check detected interactions vs billing
7. Calculate intent score
8. Generate alert if threshold exceeded
9. Extract forensic video clips
10. Build and save case file
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from .config import (
    DEFAULT_VIDEO_PATH,
    DEFAULT_POS_PATH,
    BEHAVIOR_MODEL_PATH,
    OUTPUTS_DIR
)
from .video.loader import VideoLoader
from .video.clip_extractor import ClipExtractor
from .pose.estimator import PoseEstimator
from .pose.feature_extractor import FeatureExtractor
from .pose.behavior_classifier import BehaviorClassifier
from .pos.data_loader import POSDataLoader
from .pos.mock_generator import MockPOSGenerator
from .analysis.cross_checker import CrossChecker, ProductInteraction
from .analysis.intent_scorer import IntentScorer
from .analysis.alert_generator import AlertGenerator
from .output.case_builder import CaseBuilder


def run_pipeline(
    video_path: Optional[Path] = None,
    pos_path: Optional[Path] = None,
    product_mapping: Optional[Dict[float, str]] = None
) -> dict:
    """
    Run the complete Digital Witness analysis pipeline.

    Args:
        video_path: Path to video file, or None for default
        pos_path: Path to POS JSON file, or None for default
        product_mapping: Dict mapping timestamps to product SKUs for cross-checking

    Returns:
        Dictionary with analysis results
    """
    video_path = video_path or DEFAULT_VIDEO_PATH
    pos_path = pos_path or DEFAULT_POS_PATH

    print("=" * 60)
    print("  DIGITAL WITNESS - Retail Security Analysis")
    print("  MVP Prototype")
    print("=" * 60)
    print()

    results = {}

    # Step 1: Load video
    print("[1/10] Loading video...")
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
    print("\n[2/10] Running pose estimation...")
    pose_results = []
    with video_loader:
        with PoseEstimator() as estimator:
            frame_count = 0
            for frame_num, frame in video_loader.frames(step=2):  # Process every 2nd frame
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
    print("\n[3/10] Extracting features from pose sequences...")
    feature_extractor = FeatureExtractor()
    pose_features = feature_extractor.extract_from_sequence(pose_results)
    print(f"  - Feature windows extracted: {len(pose_features)}")
    results["pose_features"] = pose_features

    # Step 4: Classify behaviors
    print("\n[4/10] Classifying behaviors...")
    try:
        classifier = BehaviorClassifier()
        behavior_events = classifier.classify_sequence(pose_features)
        behavior_events = classifier.merge_consecutive_events(behavior_events)
        print(f"  - Behavior events detected: {len(behavior_events)}")

        # Count by type
        for btype in ["normal", "pickup", "concealment", "bypass"]:
            count = sum(1 for e in behavior_events if e.behavior_type == btype)
            if count > 0:
                print(f"    - {btype}: {count}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ! Model not trained: {e}")
        print("  ! Run 'python -m src.pose.train_classifier' first")
        behavior_events = []

    results["behavior_events"] = behavior_events

    # Step 5: Load POS data
    print("\n[5/10] Loading POS transaction data...")
    try:
        pos_loader = POSDataLoader(pos_path)
        transactions = pos_loader.load()
        print(f"  - Transactions loaded: {len(transactions)}")
        total_items = sum(len(t.items) for t in transactions)
        print(f"  - Total items billed: {total_items}")
    except FileNotFoundError:
        print(f"  ! POS file not found: {pos_path}")
        print("  ! Generating mock POS data...")
        generator = MockPOSGenerator()
        mock_data = generator.generate_scenario(
            scenario_type="partial",
            base_timestamp=datetime.now(),
            video_duration=metadata.duration,
            detected_items=["ITEM001", "ITEM002", "ITEM003"]
        )
        generator.save_to_file(mock_data)
        pos_loader = POSDataLoader()
        transactions = pos_loader.load()

    results["transactions"] = transactions

    # Step 6: Cross-check interactions vs billing
    print("\n[6/10] Cross-checking detected interactions with billing...")
    cross_checker = CrossChecker()

    # Create product interactions from behavior events
    # In MVP, we simulate detected products based on pickup events
    detected_interactions = []
    if product_mapping:
        for event in behavior_events:
            if event.behavior_type == "pickup" and event.start_time in product_mapping:
                interaction = ProductInteraction(
                    sku=product_mapping[event.start_time],
                    timestamp=event.start_time,
                    interaction_type="pickup",
                    confidence=event.confidence
                )
                detected_interactions.append(interaction)
    else:
        # Demo: simulate some detected products
        demo_skus = ["ITEM001", "ITEM002", "ITEM003"]
        for i, event in enumerate(behavior_events):
            if event.behavior_type == "pickup" and i < len(demo_skus):
                interaction = ProductInteraction(
                    sku=demo_skus[i],
                    timestamp=event.start_time,
                    interaction_type="pickup",
                    confidence=event.confidence
                )
                detected_interactions.append(interaction)

    discrepancy_report = cross_checker.check_discrepancies(
        detected_interactions,
        transactions
    )
    print(f"  - Items detected: {discrepancy_report.total_detected}")
    print(f"  - Items billed: {discrepancy_report.total_billed}")
    print(f"  - Discrepancies: {discrepancy_report.discrepancy_count}")
    print(f"  - Match rate: {discrepancy_report.match_rate:.1%}")
    results["discrepancy_report"] = discrepancy_report

    # Step 7: Calculate intent score
    print("\n[7/10] Calculating intent score...")
    intent_scorer = IntentScorer()
    intent_score = intent_scorer.calculate_score(
        discrepancy_report,
        behavior_events,
        metadata.duration
    )
    print(f"  - Intent score: {intent_score.score:.2f}")
    print(f"  - Severity: {intent_score.severity.value}")
    results["intent_score"] = intent_score

    # Step 8: Generate alert if needed
    print("\n[8/10] Checking alert conditions...")
    alert_generator = AlertGenerator()

    # Get suspicious events for clip extraction
    suspicious_events = [
        {"timestamp": e.start_time, "type": e.behavior_type}
        for e in behavior_events
        if e.behavior_type in ["concealment", "bypass", "pickup"]
    ]

    # Step 9: Extract forensic clips
    print("\n[9/10] Extracting forensic video clips...")
    forensic_clips = []
    if suspicious_events:
        with video_loader:
            clip_extractor = ClipExtractor(video_loader)
            forensic_clips = clip_extractor.extract_clips_for_events(
                suspicious_events[:5]  # Limit to 5 clips
            )
        print(f"  - Clips extracted: {len(forensic_clips)}")
    else:
        print("  - No suspicious events to clip")
    results["forensic_clips"] = forensic_clips

    # Generate alert
    alert = alert_generator.generate_alert(
        intent_score,
        discrepancy_report,
        behavior_events,
        forensic_clips
    )
    if alert:
        print(f"\n  *** ALERT GENERATED ***")
        print(f"  - Alert ID: {alert.alert_id}")
        print(f"  - Severity: {alert.severity.value}")
    else:
        print("  - No alert generated (below threshold)")
    results["alert"] = alert

    # Step 10: Build case file
    print("\n[10/10] Building case file...")
    case_builder = CaseBuilder()
    case = case_builder.build_case(
        video_metadata=metadata,
        behavior_events=behavior_events,
        discrepancy_report=discrepancy_report,
        intent_score=intent_score,
        alert=alert,
        forensic_clips=forensic_clips
    )
    case_path = case_builder.save_case(case)
    print(f"  - Case ID: {case.case_id}")
    print(f"  - Saved to: {case_path}")
    results["case"] = case
    results["case_path"] = case_path

    # Final summary
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(case.summary)

    if alert:
        print("\n" + alert_generator.format_for_display(alert))

    return results


def run_demo_mode(pos_path: Optional[Path] = None) -> dict:
    """
    Run in demo mode without actual video file.
    Uses simulated data to demonstrate the pipeline.
    """
    print("\n" + "=" * 60)
    print("  DEMO MODE - Simulated Analysis")
    print("=" * 60)

    from .pose.behavior_classifier import BehaviorEvent
    from .video.loader import VideoMetadata

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
        BehaviorEvent("normal", 0.0, 10.0, 0, 300, 0.9, {"normal": 0.9, "pickup": 0.05, "concealment": 0.03, "bypass": 0.02}),
        BehaviorEvent("pickup", 10.0, 12.0, 300, 360, 0.85, {"normal": 0.1, "pickup": 0.85, "concealment": 0.03, "bypass": 0.02}),
        BehaviorEvent("normal", 12.0, 25.0, 360, 750, 0.88, {"normal": 0.88, "pickup": 0.07, "concealment": 0.03, "bypass": 0.02}),
        BehaviorEvent("pickup", 25.0, 27.0, 750, 810, 0.82, {"normal": 0.12, "pickup": 0.82, "concealment": 0.04, "bypass": 0.02}),
        BehaviorEvent("concealment", 27.0, 30.0, 810, 900, 0.75, {"normal": 0.15, "pickup": 0.05, "concealment": 0.75, "bypass": 0.05}),
        BehaviorEvent("normal", 30.0, 55.0, 900, 1650, 0.92, {"normal": 0.92, "pickup": 0.04, "concealment": 0.02, "bypass": 0.02}),
        BehaviorEvent("bypass", 55.0, 60.0, 1650, 1800, 0.7, {"normal": 0.2, "pickup": 0.05, "concealment": 0.05, "bypass": 0.7}),
    ]

    # Load or generate POS data
    print("\n[1/5] Loading POS data...")
    try:
        pos_loader = POSDataLoader(pos_path or DEFAULT_POS_PATH)
        transactions = pos_loader.load()
    except FileNotFoundError:
        generator = MockPOSGenerator()
        mock_data = generator.generate_scenario(
            scenario_type="partial",
            base_timestamp=datetime.now(),
            video_duration=60.0,
            detected_items=["ITEM001", "ITEM002", "ITEM003"]
        )
        generator.save_to_file(mock_data)
        pos_loader = POSDataLoader()
        transactions = pos_loader.load()

    print(f"  - Transactions: {len(transactions)}")

    # Cross-check
    print("\n[2/5] Cross-checking...")
    cross_checker = CrossChecker()
    detected_interactions = [
        ProductInteraction("ITEM001", 10.0, "pickup", 0.85),
        ProductInteraction("ITEM002", 25.0, "pickup", 0.82),
        ProductInteraction("ITEM003", 27.0, "pickup", 0.75),
    ]
    discrepancy_report = cross_checker.check_discrepancies(
        detected_interactions,
        transactions
    )
    print(f"  - Discrepancies: {discrepancy_report.discrepancy_count}")

    # Intent score
    print("\n[3/5] Calculating intent score...")
    intent_scorer = IntentScorer()
    intent_score = intent_scorer.calculate_score(
        discrepancy_report,
        behavior_events,
        60.0
    )
    print(f"  - Score: {intent_score.score:.2f} ({intent_score.severity.value})")

    # Alert
    print("\n[4/5] Generating alert...")
    alert_generator = AlertGenerator()
    alert = alert_generator.generate_alert(
        intent_score,
        discrepancy_report,
        behavior_events,
        []  # No clips in demo mode
    )

    # Case file
    print("\n[5/5] Building case file...")
    case_builder = CaseBuilder()
    case = case_builder.build_case(
        video_metadata=metadata,
        behavior_events=behavior_events,
        discrepancy_report=discrepancy_report,
        intent_score=intent_score,
        alert=alert,
        forensic_clips=[]
    )
    case_path = case_builder.save_case(case)
    print(f"  - Case saved: {case_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO ANALYSIS COMPLETE")
    print("=" * 60)
    print(case.summary)

    if alert:
        print("\n" + alert_generator.format_for_display(alert))

    return {
        "video_metadata": metadata,
        "behavior_events": behavior_events,
        "transactions": transactions,
        "discrepancy_report": discrepancy_report,
        "intent_score": intent_score,
        "alert": alert,
        "case": case,
        "case_path": case_path
    }


def main():
    """Main entry point."""
    # Check for command line arguments
    video_path = None
    pos_path = None

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        pos_path = Path(sys.argv[2])

    run_pipeline(video_path, pos_path)


if __name__ == "__main__":
    main()
