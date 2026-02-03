"""
Digital Witness - Main Pipeline

Deep learning analysis pipeline: YOLO -> CNN -> LSTM
Detects potential shoplifting by analyzing behavioral patterns in video.
"""
from pathlib import Path
from typing import Optional, Callable

from .config import DEFAULT_VIDEO_PATH, DEFAULT_POS_PATH
from .video.loader import VideoLoader
from .pos.data_loader import POSDataLoader
from .analysis.cross_checker import CrossChecker, DiscrepancyReport
from .analysis.alert_generator import AlertGenerator
from .output.case_builder import CaseBuilder


def run_pipeline(
    video_path: Optional[Path] = None,
    pos_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> dict:
    """
    Execute the deep learning analysis pipeline.

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
    from .models.behavior_event import BehaviorEvent
    from .analysis.intent_scorer import IntentScore

    video_path = Path(video_path) if video_path else DEFAULT_VIDEO_PATH
    pos_path = pos_path or DEFAULT_POS_PATH

    results = {}

    def update_progress(progress: float, message: str):
        if progress_callback:
            progress_callback(progress, message)

    # Initialize deep learning models
    update_progress(0.0, "Initializing models...")
    pipeline = DeepPipeline(device="auto")
    pipeline.initialize()

    # Validate video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Run deep learning analysis
    # frame_step=3 for faster processing on CPU (cloud deployment)
    # Change to frame_step=1 for maximum accuracy on GPU
    update_progress(0.05, "Processing video...")
    deep_result = pipeline.process_video(
        str(video_path),
        frame_step=3,
        progress_callback=lambda p, m: update_progress(0.05 + p * 0.55, m),
        store_frame_analyses=False
    )
    results["deep_result"] = deep_result

    # Get video metadata
    update_progress(0.60, "Loading metadata...")
    video_loader = VideoLoader(video_path)
    metadata = video_loader.metadata
    results["video_metadata"] = metadata

    # Quality analysis
    update_progress(0.65, "Analyzing quality...")
    quality_analyzer = QualityAnalyzer()
    quality_report = quality_analyzer.analyze_from_deep_result(deep_result, metadata.fps)
    results["quality_report"] = quality_report

    # Skip POS check for MVP
    update_progress(0.72, "Processing...")
    discrepancy_report = DiscrepancyReport(
        total_detected=deep_result.total_interactions,
        total_billed=0,
        matched_items=[],
        missing_from_billing=[],
        extra_in_billing=[],
        discrepancy_count=0,
        match_rate=1.0
    )
    results["discrepancy_report"] = discrepancy_report

    # Convert predictions to behavior events
    update_progress(0.78, "Processing predictions...")
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

    # Bias-aware scoring
    update_progress(0.82, "Calculating intent score...")
    bias_scorer = BiasAwareScorer()
    bias_aware_score = bias_scorer.calculate_score(
        discrepancy_report,
        behavior_events,
        metadata.duration,
        quality_report
    )
    results["bias_aware_score"] = bias_aware_score
    results["fairness_report"] = bias_aware_score.fairness_report

    # Edge case handling
    update_progress(0.85, "Checking edge cases...")
    edge_handler = EdgeCaseHandler()
    edge_report = edge_handler.analyze(behavior_events, quality_report)
    results["edge_report"] = edge_report

    # Generate alert
    update_progress(0.88, "Generating alert...")
    alert_generator = AlertGenerator()
    intent_score = IntentScore(
        score=bias_aware_score.final_score,
        severity=bias_aware_score.severity,
        components=bias_aware_score.components,
        explanation=bias_aware_score.explanation
    )
    alert = alert_generator.generate_alert(
        intent_score, discrepancy_report, behavior_events, []
    )
    if alert:
        alert.notes = f"Fairness Score: {bias_aware_score.fairness_report.overall_fairness_score:.1%}"
        if bias_aware_score.fairness_report.requires_manual_review:
            alert.notes += "\n[MANUAL REVIEW REQUIRED]"
    results["alert"] = alert
    results["intent_score"] = intent_score

    # Build case file
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
    case.notes = f"""
Deep Learning Analysis:
- Persons tracked: {deep_result.persons_tracked}
- Products tracked: {deep_result.products_tracked}
- Interactions: {deep_result.total_interactions}
- Classification: {deep_result.overall_intent} ({deep_result.overall_confidence:.1%})
"""
    case_path = case_builder.save_case(case)
    results["case"] = case
    results["case_path"] = case_path

    # Generate forensic package
    update_progress(0.94, "Generating forensic package...")
    try:
        evidence_compiler = EvidenceCompiler()
        forensic_package = evidence_compiler.compile_package(
            case_data=case.to_dict(),
            video_path=video_path,
            pose_results=None,
            behavior_events=behavior_events,
            quality_report=quality_report,
            fairness_report=bias_aware_score.fairness_report,
            generate_clips=True,
            max_clips=3
        )
        results["forensic_package"] = forensic_package
    except Exception:
        pass

    pipeline.close()
    update_progress(1.0, "Complete!")

    # Return structured results for UI
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
        "lstm_detection": {
            "classification": deep_result.overall_intent,
            "confidence": deep_result.overall_confidence,
            "is_shoplifting": deep_result.overall_intent == "shoplifting",
            "suspicious_segments": len(deep_result.suspicious_segments)
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
        "quality_analysis": {
            "reliability_score": quality_report.overall_quality_score,
            "detection_rate": quality_report.pose_detection_rate,
            "usable": quality_report.usable_for_analysis
        },
        "bias_report": {
            "overall_fairness_score": bias_aware_score.fairness_report.overall_fairness_score,
            "analysis_reliable": bias_aware_score.fairness_report.analysis_reliable,
            "requires_review": bias_aware_score.fairness_report.requires_manual_review
        },
        "alert": {
            "alert_id": alert.alert_id,
            "level": alert.severity.value,
            "message": alert.explanation
        } if alert else None,
        "case_id": case.case_id,
        "case_path": str(case_path)
    }
