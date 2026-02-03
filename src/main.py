"""
Digital Witness - Main Pipeline 

Deep learning analysis pipeline: YOLO → CNN → LSTM
Detects potential shoplifting by correlating behavioral video analysis
with POS transaction data.

Pipeline flow:
1. YOLO object detection & tracking (persons, products, interactions)
2. CNN spatial feature extraction (ResNet18)
3. LSTM temporal behavior classification
4. Bias-aware intent scoring
5. Quality & edge case analysis
6. Forensic report generation
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Callable

from .config import (
    DEFAULT_VIDEO_PATH,
    DEFAULT_POS_PATH
)
from .video.loader import VideoLoader
from .pos.data_loader import POSDataLoader
from .pos.mock_generator import MockPOSGenerator
from .analysis.cross_checker import CrossChecker, ProductInteraction
from .analysis.alert_generator import AlertGenerator
from .output.case_builder import CaseBuilder


def run_pipeline(
    video_path: Optional[Path] = None,
    pos_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> dict:
    """
    Execute the deep learning analysis pipeline: YOLO → CNN → LSTM.

    Pipeline stages:
        1-2: Object detection & tracking (YOLO)
        3-4: Feature extraction & classification (CNN + LSTM)
        5-6: Quality & bias analysis
        7-8: POS correlation & intent scoring
        9-10: Alert generation & forensic packaging

    Args:
        video_path: Source video file path
        pos_path: POS transaction JSON file path
        progress_callback: Optional callback(progress, message) for UI updates

    Returns:
        Dict containing all analysis results
    """
    from .models.deep_pipeline import DeepPipeline
    from .analysis.quality_analyzer import QualityAnalyzer
    from .analysis.bias_aware_scorer import BiasAwareScorer
    from .analysis.edge_case_handler import EdgeCaseHandler
    from .output.evidence_compiler import EvidenceCompiler

    # Use default paths if not provided
    video_path = video_path or DEFAULT_VIDEO_PATH
    pos_path = pos_path or DEFAULT_POS_PATH

    # Display system banner
    print("=" * 60)
    print("  DIGITAL WITNESS")
    print("  Deep Learning Retail Security Analysis")
    print("  YOLO -> CNN -> LSTM Pipeline")
    print("=" * 60)
    print()

    results = {}

    def update_progress(progress: float, message: str):
        if progress_callback:
            progress_callback(progress, message)
        print(f"  [{progress:.0%}] {message}")

    # ==========================================================================
    # PIPELINE STAGE 1: Model Initialization
    # Load YOLO (detection), CNN (features), and LSTM (classification) models
    # ==========================================================================
    update_progress(0.0, "Initializing deep learning models...")
    try:
        pipeline = DeepPipeline(device="auto")
        pipeline.initialize()
        print("  - YOLO detector: Ready")
        print("  - CNN feature extractor: Ready")
        print("  - LSTM classifier: Ready")
    except Exception as e:
        print(f"  ! Failed to initialize deep pipeline: {e}")
        print("  ! Running in demo mode...")
        return run_demo_mode(pos_path)

    # ==========================================================================
    # PIPELINE STAGE 2: Video Validation
    # Verify input video exists before processing
    # ==========================================================================
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  ! Video file not found: {video_path}")
        print("  ! Running in demo mode...")
        pipeline.close()
        return run_demo_mode(pos_path)

    # ==========================================================================
    # PIPELINE STAGE 3: Deep Learning Analysis (YOLO → CNN → LSTM)
    # Process video through the full deep learning pipeline:
    # - YOLO detects persons and products in each frame
    # - CNN extracts 512-dim spatial features from detections
    # - LSTM classifies temporal behavior sequences
    # ==========================================================================
    update_progress(0.05, "Processing video with deep learning...")
    try:
        deep_result = pipeline.process_video(
            str(video_path),
            frame_step=1,  # Process every frame for accurate detection
            progress_callback=lambda p, m: update_progress(0.05 + p * 0.55, m),
            store_frame_analyses=False
        )

        print(f"\n  Deep Analysis Results:")
        print(f"  - Frames processed: {deep_result.processed_frames}")
        print(f"  - Persons tracked: {deep_result.persons_tracked}")
        print(f"  - Products tracked: {deep_result.products_tracked}")
        print(f"  - Interactions detected: {deep_result.total_interactions}")
        print(f"  - Overall intent: {deep_result.overall_intent}")
        print(f"  - Confidence: {deep_result.overall_confidence:.2%}")

        results["deep_result"] = deep_result

    except Exception as e:
        print(f"  ! Error in deep pipeline: {e}")
        pipeline.close()
        return run_demo_mode(pos_path)

    # ==========================================================================
    # PIPELINE STAGE 4-5: Metadata & Quality Analysis
    # Extract video properties and assess quality for reliability scoring
    # ==========================================================================
    update_progress(0.60, "Loading video metadata...")
    video_loader = VideoLoader(video_path)
    metadata = video_loader.metadata
    results["video_metadata"] = metadata

    update_progress(0.65, "Assessing video quality...")
    quality_analyzer = QualityAnalyzer()
    quality_report = quality_analyzer.analyze_from_deep_result(deep_result, metadata.fps)
    print(f"\n  Quality Analysis:")
    print(f"  - Detection rate: {quality_report.pose_detection_rate:.1%}")
    print(f"  - Quality score: {quality_report.overall_quality_score:.2f}")
    print(f"  - Usable: {'Yes' if quality_report.usable_for_analysis else 'No'}")
    results["quality_report"] = quality_report

    # ==========================================================================
    # PIPELINE STAGE 6-7: POS Data Loading (OPTIONAL for MVP)
    # For MVP, we skip POS and rely purely on video behavior analysis
    # ==========================================================================
    update_progress(0.72, "Skipping POS check (MVP mode)...")
    transactions = []
    results["transactions"] = transactions

    # Create empty discrepancy report for MVP (no POS cross-check)
    from .analysis.cross_checker import DiscrepancyReport
    discrepancy_report = DiscrepancyReport(
        total_detected=deep_result.total_interactions,
        total_billed=0,
        matched_items=[],
        missing_from_billing=[],
        extra_in_billing=[],
        discrepancy_count=0,
        match_rate=1.0  # No discrepancy when no POS data
    )
    print(f"  - MVP Mode: POS cross-check skipped")
    results["discrepancy_report"] = discrepancy_report

    # Step 9: Convert deep predictions to behavior events
    update_progress(0.78, "Processing behavior predictions...")
    from .models.behavior_event import BehaviorEvent

    behavior_events = []
    for pred in deep_result.intent_predictions:
        event = BehaviorEvent(
            behavior_type=pred.intent_class,
            start_time=pred.start_time or 0.0,
            end_time=pred.end_time or 0.0,
            start_frame=int((pred.start_time or 0) * metadata.fps),
            end_frame=int((pred.end_time or 0) * metadata.fps),
            confidence=pred.confidence,
            probabilities=pred.class_probabilities
        )
        behavior_events.append(event)
    results["behavior_events"] = behavior_events

    # ==========================================================================
    # PIPELINE STAGE 10: Bias-Aware Intent Scoring
    # Calculate risk score with fairness adjustments to prevent bias
    # Score formula: (discrepancy*0.4) + (concealment*0.3) + (bypass*0.2) + (duration*0.1)
    # Adjustments applied when bias indicators are detected
    # ==========================================================================
    update_progress(0.82, "Calculating bias-aware intent score...")
    bias_scorer = BiasAwareScorer()
    bias_aware_score = bias_scorer.calculate_score(
        discrepancy_report,
        behavior_events,
        metadata.duration,
        quality_report
    )

    print(f"\n  Bias-Aware Intent Assessment:")
    print(f"  - Raw score: {bias_aware_score.raw_score:.2f}")
    print(f"  - Adjusted score: {bias_aware_score.adjusted_score:.2f}")
    print(f"  - Bias adjustment: {bias_aware_score.bias_adjustment_factor:.2f}")
    print(f"  - Confidence: {bias_aware_score.confidence_level}")
    print(f"  - Fairness score: {bias_aware_score.fairness_report.overall_fairness_score:.1%}")

    results["bias_aware_score"] = bias_aware_score
    results["fairness_report"] = bias_aware_score.fairness_report

    # Step 11: Edge case handling
    update_progress(0.85, "Checking for edge cases...")
    edge_handler = EdgeCaseHandler()
    edge_report = edge_handler.analyze(behavior_events, quality_report)

    if edge_report.requires_manual_review:
        print(f"  - Edge cases detected: {edge_report.total_flags}")
        print(f"  - Manual review: REQUIRED")
    results["edge_report"] = edge_report

    # ==========================================================================
    # PIPELINE STAGE 12: Alert Generation
    # Generate advisory alert if score exceeds threshold (default: 0.5)
    # IMPORTANT: All alerts require human review before action
    # ==========================================================================
    update_progress(0.88, "Generating alert...")
    alert_generator = AlertGenerator()

    # Convert bias-aware score to standard intent score for alert
    from .analysis.intent_scorer import IntentScore
    intent_score = IntentScore(
        score=bias_aware_score.final_score,
        severity=bias_aware_score.severity,
        components=bias_aware_score.components,
        explanation=bias_aware_score.explanation
    )

    alert = alert_generator.generate_alert(
        intent_score,
        discrepancy_report,
        behavior_events,
        []  # Clips generated in forensic package
    )

    if alert:
        # Add fairness info to alert
        alert.notes = (alert.notes or "") + f"\nFairness Score: {bias_aware_score.fairness_report.overall_fairness_score:.1%}"
        if bias_aware_score.fairness_report.requires_manual_review:
            alert.notes += "\n[BIAS INDICATORS DETECTED - MANUAL REVIEW REQUIRED]"
        print(f"\n  *** ALERT GENERATED ***")
        print(f"  - Severity: {alert.severity.value}")
    else:
        print("  - No alert generated (below threshold)")

    results["alert"] = alert
    results["intent_score"] = intent_score

    # ==========================================================================
    # PIPELINE STAGE 13: Case File Generation
    # Create JSON audit trail with all evidence for review and archival
    # ==========================================================================
    update_progress(0.91, "Building case file...")
    case_builder = CaseBuilder()
    case = case_builder.build_case(
        video_metadata=metadata,
        behavior_events=behavior_events,
        discrepancy_report=discrepancy_report,
        intent_score=intent_score,
        alert=alert,
        forensic_clips=[]
    )

    # Add deep learning specific info to case
    case.notes = f"""
Deep Learning Analysis Results:
- Pipeline: YOLO → CNN → LSTM
- Persons tracked: {deep_result.persons_tracked}
- Products tracked: {deep_result.products_tracked}
- Interactions: {deep_result.total_interactions}
- Suspicious segments: {len(deep_result.suspicious_segments)}

Quality Analysis:
- Detection rate: {quality_report.pose_detection_rate:.1%}
- Quality score: {quality_report.overall_quality_score:.2f}

Fairness Assessment:
- Fairness score: {bias_aware_score.fairness_report.overall_fairness_score:.1%}
- Bias risk: {bias_aware_score.fairness_report.bias_metrics.overall_bias_risk.value}
- Analysis reliable: {bias_aware_score.fairness_report.analysis_reliable}
"""

    case_path = case_builder.save_case(case)
    results["case"] = case
    results["case_path"] = case_path

    # Step 14: Generate forensic package
    # Compile all evidence into a self-contained forensic package for case review
    update_progress(0.94, "Generating forensic package...")
    try:
        evidence_compiler = EvidenceCompiler()
        forensic_package = evidence_compiler.compile_package(
            case_data=case.to_dict(),
            video_path=video_path,
            pose_results=None,  # Pose results not used in deep learning pipeline
            behavior_events=behavior_events,
            quality_report=quality_report,
            fairness_report=bias_aware_score.fairness_report,
            generate_clips=True,
            max_clips=3
        )
        print(f"  - Forensic package: {forensic_package.package_dir}")
        results["forensic_package"] = forensic_package
    except Exception as e:
        print(f"  - Forensic package generation skipped: {e}")

    # Cleanup
    pipeline.close()

    update_progress(1.0, "Analysis complete!")

    # Final summary
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Case ID: {case.case_id}")
    print(f"  Saved to: {case_path}")
    print(f"\n{bias_aware_score.explanation}")

    if alert:
        print("\n" + alert_generator.format_for_display(alert))

    # Return results in format expected by Streamlit app
    return {
        "success": True,
        "video_metadata": {
            "filename": metadata.path.name,
            "duration": metadata.duration,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": metadata.frame_count
        },
        "detections": {
            "persons_tracked": deep_result.persons_tracked,
            "products_detected": deep_result.products_tracked,
            "interactions": deep_result.total_interactions,
            "frames_processed": deep_result.processed_frames
        },
        "behavior_events": [
            {
                "behavior_type": e.behavior_type,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "confidence": e.confidence,
                "probabilities": e.probabilities
            }
            for e in behavior_events
        ],
        "intent_score": {
            "score": intent_score.score,
            "severity": intent_score.severity.value,
            "components": intent_score.components,
            "explanation": intent_score.explanation
        },
        "discrepancy_report": {
            "total_detected": discrepancy_report.total_detected,
            "total_billed": discrepancy_report.total_billed,
            "discrepancy_count": discrepancy_report.discrepancy_count,
            "match_rate": discrepancy_report.match_rate,
            "missing_from_billing": discrepancy_report.missing_from_billing
        },
        "quality_analysis": {
            "reliability_score": quality_report.overall_quality_score,
            "detection_rate": quality_report.pose_detection_rate,
            "usable": quality_report.usable_for_analysis,
            "issues": quality_report.quality_issues if hasattr(quality_report, 'quality_issues') else []
        },
        "bias_report": {
            "overall_fairness_score": bias_aware_score.fairness_report.overall_fairness_score,
            "analysis_reliable": bias_aware_score.fairness_report.analysis_reliable,
            "requires_review": bias_aware_score.fairness_report.requires_manual_review,
            "flags": [str(f) for f in bias_aware_score.fairness_report.bias_metrics.flags] if hasattr(bias_aware_score.fairness_report.bias_metrics, 'flags') else []
        },
        "alert": {
            "alert_id": alert.alert_id,
            "level": alert.severity.value,
            "message": alert.explanation
        } if alert else None,
        "case_id": case.case_id,
        "case_path": str(case_path),
        # Keep raw results for advanced use
        "_raw": results
    }


def run_demo_mode(pos_path: Optional[Path] = None) -> dict:
    """
    Run in demo mode without actual video file.
    Uses simulated data to demonstrate the pipeline.
    """
    print("\n" + "=" * 60)
    print("  DEMO MODE - Simulated Analysis")
    print("=" * 60)

    from .models.behavior_event import BehaviorEvent
    from .video.loader import VideoMetadata
    from .analysis.intent_scorer import IntentScorer

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

    # Simulate behavior events (as if from LSTM)
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
        []
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
    case.notes = "DEMO MODE - Simulated data for demonstration purposes"
    case_path = case_builder.save_case(case)
    print(f"  - Case saved: {case_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO ANALYSIS COMPLETE")
    print("=" * 60)
    print(case.summary)

    if alert:
        print("\n" + alert_generator.format_for_display(alert))

    # Return results in format expected by Streamlit app
    return {
        "success": True,
        "demo_mode": True,
        "video_metadata": {
            "filename": metadata.path.name,
            "duration": metadata.duration,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": metadata.frame_count
        },
        "detections": {
            "persons_tracked": 1,
            "products_detected": 3,
            "interactions": len(detected_interactions),
            "frames_processed": metadata.frame_count
        },
        "behavior_events": [
            {
                "behavior_type": e.behavior_type,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "confidence": e.confidence,
                "probabilities": e.probabilities
            }
            for e in behavior_events
        ],
        "intent_score": {
            "score": intent_score.score,
            "severity": intent_score.severity.value,
            "components": intent_score.components,
            "explanation": intent_score.explanation
        },
        "discrepancy_report": {
            "total_detected": discrepancy_report.total_detected,
            "total_billed": discrepancy_report.total_billed,
            "discrepancy_count": discrepancy_report.discrepancy_count,
            "match_rate": discrepancy_report.match_rate,
            "missing_from_billing": discrepancy_report.missing_from_billing
        },
        "quality_analysis": {
            "reliability_score": 0.85,
            "detection_rate": 0.90,
            "usable": True,
            "issues": []
        },
        "bias_report": {
            "overall_fairness_score": 0.90,
            "analysis_reliable": True,
            "requires_review": False,
            "flags": []
        },
        "alert": {
            "alert_id": alert.alert_id,
            "level": alert.severity.value,
            "message": alert.explanation
        } if alert else None,
        "case_id": case.case_id,
        "case_path": str(case_path)
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
