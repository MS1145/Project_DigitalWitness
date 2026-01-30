"""
PDF report generation for Digital Witness.

Generates professional forensic reports in PDF format including:
- Case summary
- Timeline of events
- Evidence screenshots
- Behavior analysis
- Fairness assessment
- Recommendations

Uses reportlab for PDF generation.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from ..config import FORENSIC_REPORTS_DIR


@dataclass
class ReportSection:
    """A section of the PDF report."""
    title: str
    content: str
    images: List[Path] = None
    table_data: Optional[List[List[str]]] = None


@dataclass
class ForensicReport:
    """Complete forensic report package."""
    case_id: str
    generated_at: str
    pdf_path: Path
    summary: str
    sections: List[str]
    page_count: int


class ReportGenerator:
    """
    Generates PDF forensic reports for case review.

    Creates comprehensive, professional reports that document
    all analysis results with supporting evidence.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or FORENSIC_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        case_data: Dict[str, Any],
        screenshots: List[Path] = None,
        include_timeline: bool = True,
        include_fairness: bool = True
    ) -> ForensicReport:
        """
        Generate a complete PDF forensic report.

        Args:
            case_data: Case file data dictionary
            screenshots: List of evidence screenshot paths
            include_timeline: Include behavior timeline section
            include_fairness: Include fairness assessment section

        Returns:
            ForensicReport object with PDF path
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image, PageBreak, ListFlowable, ListItem
            )
        except ImportError:
            # Fallback to text report if reportlab not available
            return self._generate_text_report(case_data)

        case_id = case_data.get("case_id", "UNKNOWN")
        pdf_path = self.output_dir / f"{case_id}_report.pdf"

        # Create document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20
        ))
        styles.add(ParagraphStyle(
            name='SubSection',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=6
        ))

        # Build content
        story = []
        sections = []

        # Title Page
        story.append(Paragraph("DIGITAL WITNESS", styles['CustomTitle']))
        story.append(Paragraph("Forensic Analysis Report", styles['Heading2']))
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Case ID: {case_id}", styles['Normal']))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 20))

        # Disclaimer
        story.append(Paragraph(
            "<b>IMPORTANT:</b> This report is for advisory purposes only. "
            "All findings require human verification before any action is taken. "
            "This system does NOT determine guilt.",
            styles['Normal']
        ))
        story.append(PageBreak())
        sections.append("Title Page")

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['SectionTitle']))
        summary = self._generate_summary(case_data)
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, 20))
        sections.append("Executive Summary")

        # Video Information
        story.append(Paragraph("Video Information", styles['SectionTitle']))
        video_meta = case_data.get("video_metadata", {})
        video_info = [
            ["Property", "Value"],
            ["File", str(video_meta.get("path", "N/A"))],
            ["Duration", f"{video_meta.get('duration', 0):.1f} seconds"],
            ["Resolution", f"{video_meta.get('width', 0)}x{video_meta.get('height', 0)}"],
            ["FPS", str(video_meta.get("fps", 0))],
        ]
        story.append(self._create_table(video_info))
        story.append(Spacer(1, 20))
        sections.append("Video Information")

        # Intent Score
        story.append(Paragraph("Intent Assessment", styles['SectionTitle']))
        intent = case_data.get("intent_score", {})
        story.append(Paragraph(
            f"<b>Intent Score:</b> {intent.get('score', 0):.2f}",
            styles['Normal']
        ))
        story.append(Paragraph(
            f"<b>Severity:</b> {intent.get('severity', 'UNKNOWN')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))

        # Component breakdown
        components = intent.get("components", {})
        if components:
            comp_data = [["Component", "Score", "Weight", "Contribution"]]
            for name, comp in components.items():
                comp_data.append([
                    name.title(),
                    f"{comp.get('score', 0):.2f}",
                    f"{comp.get('weight', 0):.1%}",
                    f"{comp.get('contribution', 0):.3f}"
                ])
            story.append(self._create_table(comp_data))
        story.append(Spacer(1, 20))
        sections.append("Intent Assessment")

        # Behavior Timeline
        if include_timeline:
            story.append(Paragraph("Behavior Timeline", styles['SectionTitle']))
            timeline = case_data.get("behavior_timeline", [])
            if timeline:
                timeline_data = [["Time Range", "Behavior", "Confidence"]]
                for event in timeline[:20]:  # Limit to 20 events
                    time_range = f"{event.get('start_time', 0):.1f}s - {event.get('end_time', 0):.1f}s"
                    behavior = event.get("behavior_type", "unknown")
                    conf = event.get("confidence", 0)
                    timeline_data.append([time_range, behavior, f"{conf:.1%}"])
                story.append(self._create_table(timeline_data))
            else:
                story.append(Paragraph("No behavior events recorded.", styles['Normal']))
            story.append(Spacer(1, 20))
            sections.append("Behavior Timeline")

        # Discrepancy Report
        story.append(Paragraph("POS Discrepancy Report", styles['SectionTitle']))
        discrepancy = case_data.get("discrepancy_report", {})
        disc_data = [
            ["Metric", "Value"],
            ["Items Detected", str(discrepancy.get("total_detected", 0))],
            ["Items Billed", str(discrepancy.get("total_billed", 0))],
            ["Discrepancies", str(discrepancy.get("discrepancy_count", 0))],
            ["Match Rate", f"{discrepancy.get('match_rate', 0):.1%}"],
        ]
        story.append(self._create_table(disc_data))

        missing = discrepancy.get("missing_from_billing", [])
        if missing:
            story.append(Spacer(1, 10))
            story.append(Paragraph("<b>Missing from billing:</b>", styles['Normal']))
            items = ListFlowable(
                [ListItem(Paragraph(item, styles['Normal'])) for item in missing],
                bulletType='bullet'
            )
            story.append(items)
        story.append(Spacer(1, 20))
        sections.append("POS Discrepancy Report")

        # Fairness Assessment
        if include_fairness:
            story.append(Paragraph("Fairness Assessment", styles['SectionTitle']))
            fairness = case_data.get("fairness_report", {})
            if fairness:
                story.append(Paragraph(
                    f"<b>Overall Fairness Score:</b> {fairness.get('overall_fairness_score', 0):.1%}",
                    styles['Normal']
                ))
                story.append(Paragraph(
                    f"<b>Analysis Reliable:</b> {'Yes' if fairness.get('analysis_reliable') else 'No'}",
                    styles['Normal']
                ))

                flags = fairness.get("flagged_issues", [])
                if flags:
                    story.append(Spacer(1, 10))
                    story.append(Paragraph("<b>Flagged Issues:</b>", styles['Normal']))
                    items = ListFlowable(
                        [ListItem(Paragraph(flag, styles['Normal'])) for flag in flags],
                        bulletType='bullet'
                    )
                    story.append(items)
            else:
                story.append(Paragraph("Fairness assessment not available.", styles['Normal']))
            story.append(Spacer(1, 20))
            sections.append("Fairness Assessment")

        # Evidence Screenshots
        if screenshots:
            story.append(PageBreak())
            story.append(Paragraph("Evidence Screenshots", styles['SectionTitle']))
            for i, screenshot in enumerate(screenshots[:10]):  # Limit to 10
                if screenshot.exists():
                    try:
                        img = Image(str(screenshot), width=5*inch, height=3*inch)
                        story.append(img)
                        story.append(Paragraph(f"Screenshot {i+1}: {screenshot.name}", styles['Normal']))
                        story.append(Spacer(1, 20))
                    except:
                        pass
            sections.append("Evidence Screenshots")

        # Alert Information
        alert = case_data.get("alert", {})
        if alert:
            story.append(PageBreak())
            story.append(Paragraph("Alert Details", styles['SectionTitle']))
            story.append(Paragraph(f"<b>Alert ID:</b> {alert.get('alert_id', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Severity:</b> {alert.get('severity', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Requires Human Review:</b> Yes", styles['Normal']))
            story.append(Spacer(1, 20))
            sections.append("Alert Details")

        # Recommendations
        story.append(Paragraph("Recommendations", styles['SectionTitle']))
        recommendations = [
            "All findings must be verified by a human operator before any action",
            "Review video evidence to confirm system observations",
            "Cross-reference with additional camera angles if available",
            "Consider environmental factors that may affect detection accuracy",
            "Document all review decisions for audit purposes"
        ]
        items = ListFlowable(
            [ListItem(Paragraph(rec, styles['Normal'])) for rec in recommendations],
            bulletType='bullet'
        )
        story.append(items)
        sections.append("Recommendations")

        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            "--- END OF REPORT ---",
            styles['Normal']
        ))
        story.append(Paragraph(
            f"Digital Witness Forensic Analysis System - {datetime.now().year}",
            styles['Normal']
        ))

        # Build PDF
        doc.build(story)

        return ForensicReport(
            case_id=case_id,
            generated_at=datetime.now().isoformat(),
            pdf_path=pdf_path,
            summary=summary,
            sections=sections,
            page_count=len(story) // 10 + 1  # Rough estimate
        )

    def _generate_summary(self, case_data: Dict) -> str:
        """Generate executive summary text."""
        intent = case_data.get("intent_score", {})
        discrepancy = case_data.get("discrepancy_report", {})
        alert = case_data.get("alert")

        score = intent.get("score", 0)
        severity = intent.get("severity", "UNKNOWN")
        disc_count = discrepancy.get("discrepancy_count", 0)

        summary_parts = []

        summary_parts.append(
            f"This analysis examined video footage and point-of-sale data "
            f"to assess potential shoplifting behavior."
        )

        if alert:
            summary_parts.append(
                f"The system generated a {severity} severity alert "
                f"with an intent score of {score:.2f}."
            )
        else:
            summary_parts.append(
                f"No alert was generated. Intent score: {score:.2f} ({severity})."
            )

        if disc_count > 0:
            summary_parts.append(
                f"{disc_count} item(s) were detected but not found in billing records."
            )

        summary_parts.append(
            "All findings require human verification before any action is taken."
        )

        return " ".join(summary_parts)

    def _create_table(self, data: List[List[str]]):
        """Create a formatted table."""
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return table

    def _generate_text_report(self, case_data: Dict) -> ForensicReport:
        """Fallback text report when reportlab not available."""
        case_id = case_data.get("case_id", "UNKNOWN")
        txt_path = self.output_dir / f"{case_id}_report.txt"

        lines = [
            "=" * 60,
            "DIGITAL WITNESS - FORENSIC ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Case ID: {case_id}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "IMPORTANT: This report is for advisory purposes only.",
            "All findings require human verification.",
            "",
            "-" * 60,
            "SUMMARY",
            "-" * 60,
            self._generate_summary(case_data),
            "",
        ]

        # Add intent score
        intent = case_data.get("intent_score", {})
        lines.extend([
            "-" * 60,
            "INTENT SCORE",
            "-" * 60,
            f"Score: {intent.get('score', 0):.2f}",
            f"Severity: {intent.get('severity', 'UNKNOWN')}",
            ""
        ])

        # Add discrepancy
        disc = case_data.get("discrepancy_report", {})
        lines.extend([
            "-" * 60,
            "DISCREPANCY REPORT",
            "-" * 60,
            f"Items Detected: {disc.get('total_detected', 0)}",
            f"Items Billed: {disc.get('total_billed', 0)}",
            f"Discrepancies: {disc.get('discrepancy_count', 0)}",
            f"Match Rate: {disc.get('match_rate', 0):.1%}",
            ""
        ])

        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])

        with open(txt_path, 'w') as f:
            f.write("\n".join(lines))

        return ForensicReport(
            case_id=case_id,
            generated_at=datetime.now().isoformat(),
            pdf_path=txt_path,
            summary=self._generate_summary(case_data),
            sections=["Summary", "Intent Score", "Discrepancy Report"],
            page_count=1
        )
