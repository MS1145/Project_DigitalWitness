"""
Cross-checker module for video-to-POS reconciliation.

Compares products detected via video analysis (pickups) against
items actually billed in POS transactions. The discrepancy between
these sets is a primary indicator of potential shoplifting.
"""
from dataclasses import dataclass
from typing import List, Dict, Set

from ..models.behavior_event import BehaviorEvent
from ..pos.data_loader import Transaction


@dataclass
class ProductInteraction:
    """A product pickup detected from video analysis."""
    sku: str                  # Product identifier
    timestamp: float          # When the pickup occurred
    interaction_type: str     # "pickup", "return", etc.
    confidence: float         # Detection confidence


@dataclass
class DiscrepancyReport:
    """
    Reconciliation result between video and POS data.

    Key metric: missing_from_billing indicates items picked up
    but not paid for - the core shoplifting indicator.
    """
    missing_from_billing: List[str]  # Picked up but not billed (suspicious)
    extra_in_billing: List[str]      # Billed but not seen (camera blind spot)
    matched_items: List[str]         # Successfully reconciled
    total_detected: int
    total_billed: int
    discrepancy_count: int           # len(missing_from_billing)
    match_rate: float                # Fraction of detected items that were billed


class CrossChecker:
    """Reconciles video-detected product interactions with POS billing."""

    def __init__(self):
        """Initialize cross-checker."""
        pass

    def check_discrepancies(
        self,
        detected_interactions: List[ProductInteraction],
        transactions: List[Transaction]
    ) -> DiscrepancyReport:
        """
        Perform set-based reconciliation between video and POS.

        Uses set operations to find:
        - detected - billed = missing (potential theft)
        - billed - detected = extra (camera missed it)
        - detected & billed = matched (normal purchase)
        """
        detected_skus = self._get_detected_pickup_skus(detected_interactions)
        billed_skus = self._get_billed_skus(transactions)

        # Set operations for reconciliation
        missing_from_billing = list(detected_skus - billed_skus)
        extra_in_billing = list(billed_skus - detected_skus)
        matched_items = list(detected_skus & billed_skus)

        # Calculate metrics
        total_detected = len(detected_skus)
        total_billed = len(billed_skus)
        discrepancy_count = len(missing_from_billing)

        # Match rate: what fraction of detected items were billed
        if total_detected > 0:
            match_rate = len(matched_items) / total_detected
        else:
            match_rate = 1.0  # No items detected = no discrepancy

        return DiscrepancyReport(
            missing_from_billing=missing_from_billing,
            extra_in_billing=extra_in_billing,
            matched_items=matched_items,
            total_detected=total_detected,
            total_billed=total_billed,
            discrepancy_count=discrepancy_count,
            match_rate=match_rate
        )

    def check_from_behavior_events(
        self,
        behavior_events: List[BehaviorEvent],
        transactions: List[Transaction],
        product_mapping: Dict[float, str]
    ) -> DiscrepancyReport:
        """
        Check discrepancies using behavior events and a time-to-product mapping.

        For MVP, we use a simplified approach where pickup events are mapped
        to products based on their timestamp.

        Args:
            behavior_events: List of classified behavior events
            transactions: List of POS transactions
            product_mapping: Dict mapping timestamps to product SKUs

        Returns:
            DiscrepancyReport with comparison results
        """
        # Convert behavior events to product interactions
        interactions = []

        for event in behavior_events:
            if event.behavior_type == "pickup":
                # Find product for this timestamp
                sku = self._find_product_for_timestamp(
                    event.start_time,
                    product_mapping
                )
                if sku:
                    interaction = ProductInteraction(
                        sku=sku,
                        timestamp=event.start_time,
                        interaction_type="pickup",
                        confidence=event.confidence
                    )
                    interactions.append(interaction)

        return self.check_discrepancies(interactions, transactions)

    def _get_detected_pickup_skus(
        self,
        interactions: List[ProductInteraction]
    ) -> Set[str]:
        """Get unique SKUs from pickup interactions."""
        return {
            i.sku for i in interactions
            if i.interaction_type == "pickup"
        }

    def _get_billed_skus(
        self,
        transactions: List[Transaction]
    ) -> Set[str]:
        """Get unique SKUs from all transactions."""
        skus = set()
        for txn in transactions:
            for item in txn.items:
                skus.add(item.sku)
        return skus

    def _find_product_for_timestamp(
        self,
        timestamp: float,
        product_mapping: Dict[float, str],
        tolerance: float = 2.0
    ) -> str:
        """
        Find product SKU for a given timestamp.

        Args:
            timestamp: Event timestamp
            product_mapping: Dict mapping timestamps to SKUs
            tolerance: Time tolerance in seconds

        Returns:
            Product SKU or empty string if not found
        """
        for mapping_time, sku in product_mapping.items():
            if abs(mapping_time - timestamp) <= tolerance:
                return sku
        return ""

    def generate_summary(self, report: DiscrepancyReport) -> str:
        """
        Generate a human-readable summary of discrepancies.

        Args:
            report: DiscrepancyReport to summarize

        Returns:
            Summary string
        """
        lines = [
            "=== Discrepancy Report ===",
            f"Detected items: {report.total_detected}",
            f"Billed items: {report.total_billed}",
            f"Match rate: {report.match_rate:.1%}",
            ""
        ]

        if report.missing_from_billing:
            lines.append("Items NOT billed (potential discrepancy):")
            for sku in report.missing_from_billing:
                lines.append(f"  - {sku}")
        else:
            lines.append("All detected items were billed.")

        if report.extra_in_billing:
            lines.append("\nItems billed but not detected:")
            for sku in report.extra_in_billing:
                lines.append(f"  - {sku}")

        if report.matched_items:
            lines.append(f"\nMatched items: {', '.join(report.matched_items)}")

        return "\n".join(lines)
