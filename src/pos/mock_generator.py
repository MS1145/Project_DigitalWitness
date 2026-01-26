"""
Mock POS transaction data generator for Digital Witness MVP.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from ..config import POS_DATA_DIR


@dataclass
class POSItem:
    """A single item in a POS transaction."""
    sku: str
    name: str
    quantity: int
    price: float


@dataclass
class POSTransaction:
    """A complete POS transaction."""
    transaction_id: str
    timestamp: str  # ISO format
    items: List[Dict]
    total: float
    payment_method: str


# Sample product catalog
PRODUCT_CATALOG = [
    {"sku": "ITEM001", "name": "Snack Bar", "price": 2.99},
    {"sku": "ITEM002", "name": "Soda Bottle", "price": 1.99},
    {"sku": "ITEM003", "name": "Chocolate Box", "price": 5.99},
    {"sku": "ITEM004", "name": "Energy Drink", "price": 3.49},
    {"sku": "ITEM005", "name": "Chips Bag", "price": 2.49},
    {"sku": "ITEM006", "name": "Candy Pack", "price": 1.49},
    {"sku": "ITEM007", "name": "Gum Pack", "price": 0.99},
    {"sku": "ITEM008", "name": "Protein Bar", "price": 3.99},
    {"sku": "ITEM009", "name": "Water Bottle", "price": 1.29},
    {"sku": "ITEM010", "name": "Coffee Can", "price": 2.79},
]

PAYMENT_METHODS = ["card", "cash", "mobile"]


class MockPOSGenerator:
    """Generates mock POS transaction data for testing."""

    def __init__(self, catalog: Optional[List[Dict]] = None):
        """
        Initialize generator.

        Args:
            catalog: Product catalog to use, or None for default
        """
        self.catalog = catalog or PRODUCT_CATALOG

    def generate_transaction(
        self,
        transaction_id: str,
        timestamp: datetime,
        items: List[Dict],
        payment_method: Optional[str] = None
    ) -> POSTransaction:
        """
        Generate a single transaction.

        Args:
            transaction_id: Unique transaction ID
            timestamp: Transaction timestamp
            items: List of items with sku and quantity
            payment_method: Payment method, or random if None

        Returns:
            POSTransaction object
        """
        # Build items list with prices
        transaction_items = []
        total = 0.0

        for item in items:
            product = next(
                (p for p in self.catalog if p['sku'] == item['sku']),
                None
            )
            if product:
                quantity = item.get('quantity', 1)
                item_total = product['price'] * quantity
                transaction_items.append({
                    "sku": product['sku'],
                    "name": product['name'],
                    "quantity": quantity,
                    "price": product['price']
                })
                total += item_total

        return POSTransaction(
            transaction_id=transaction_id,
            timestamp=timestamp.isoformat(),
            items=transaction_items,
            total=round(total, 2),
            payment_method=payment_method or random.choice(PAYMENT_METHODS)
        )

    def generate_random_transaction(
        self,
        transaction_id: str,
        base_timestamp: datetime,
        min_items: int = 1,
        max_items: int = 5
    ) -> POSTransaction:
        """
        Generate a random transaction.

        Args:
            transaction_id: Unique transaction ID
            base_timestamp: Base timestamp (adds random offset)
            min_items: Minimum number of items
            max_items: Maximum number of items

        Returns:
            POSTransaction object
        """
        # Random number of items
        n_items = random.randint(min_items, max_items)

        # Select random products
        selected_products = random.sample(self.catalog, min(n_items, len(self.catalog)))

        items = [
            {"sku": p['sku'], "quantity": random.randint(1, 2)}
            for p in selected_products
        ]

        # Add random time offset (0-60 seconds)
        timestamp = base_timestamp + timedelta(seconds=random.randint(0, 60))

        return self.generate_transaction(
            transaction_id=transaction_id,
            timestamp=timestamp,
            items=items
        )

    def generate_scenario(
        self,
        scenario_type: str,
        base_timestamp: datetime,
        video_duration: float,
        detected_items: List[str]
    ) -> Dict:
        """
        Generate POS data for a specific scenario.

        Args:
            scenario_type: "complete", "partial", or "none"
            base_timestamp: Video start timestamp
            video_duration: Video duration in seconds
            detected_items: List of SKUs detected in video

        Returns:
            Dictionary with transactions list
        """
        transactions = []

        if scenario_type == "complete":
            # All detected items are billed
            checkout_time = base_timestamp + timedelta(seconds=video_duration * 0.8)
            items = [{"sku": sku, "quantity": 1} for sku in detected_items]
            txn = self.generate_transaction(
                transaction_id="TXN001",
                timestamp=checkout_time,
                items=items
            )
            transactions.append(asdict(txn))

        elif scenario_type == "partial":
            # Only some items are billed
            checkout_time = base_timestamp + timedelta(seconds=video_duration * 0.8)
            # Bill only first half of items
            billed_items = detected_items[:len(detected_items) // 2]
            items = [{"sku": sku, "quantity": 1} for sku in billed_items]
            if items:
                txn = self.generate_transaction(
                    transaction_id="TXN001",
                    timestamp=checkout_time,
                    items=items
                )
                transactions.append(asdict(txn))

        elif scenario_type == "none":
            # No billing at all
            pass

        return {"transactions": transactions}

    def save_to_file(
        self,
        data: Dict,
        filename: str = "transactions.json",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save transaction data to JSON file.

        Args:
            data: Transaction data dictionary
            filename: Output filename
            output_dir: Output directory, or default POS_DATA_DIR

        Returns:
            Path to saved file
        """
        output_dir = output_dir or POS_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path


def generate_sample_pos_data():
    """Generate sample POS data files for testing."""
    generator = MockPOSGenerator()
    base_time = datetime.now()

    # Complete billing scenario
    complete_data = generator.generate_scenario(
        scenario_type="complete",
        base_timestamp=base_time,
        video_duration=60.0,
        detected_items=["ITEM001", "ITEM002", "ITEM003"]
    )
    generator.save_to_file(complete_data, "complete_billing.json")

    # Partial billing scenario
    partial_data = generator.generate_scenario(
        scenario_type="partial",
        base_timestamp=base_time,
        video_duration=60.0,
        detected_items=["ITEM001", "ITEM002", "ITEM003", "ITEM004"]
    )
    generator.save_to_file(partial_data, "partial_billing.json")

    # No billing scenario
    none_data = generator.generate_scenario(
        scenario_type="none",
        base_timestamp=base_time,
        video_duration=60.0,
        detected_items=["ITEM001", "ITEM002"]
    )
    generator.save_to_file(none_data, "no_billing.json")

    print("Generated sample POS data files in:", POS_DATA_DIR)


if __name__ == "__main__":
    generate_sample_pos_data()
