"""
POS data loader for Digital Witness.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..config import DEFAULT_POS_PATH


@dataclass
class TransactionItem:
    """A single item in a transaction."""
    sku: str
    name: str
    quantity: int
    price: float


@dataclass
class Transaction:
    """A parsed POS transaction."""
    transaction_id: str
    timestamp: datetime
    items: List[TransactionItem]
    total: float
    payment_method: str

    def get_item_skus(self) -> List[str]:
        """Get list of all item SKUs in transaction."""
        skus = []
        for item in self.items:
            skus.extend([item.sku] * item.quantity)
        return skus

    def has_item(self, sku: str) -> bool:
        """Check if transaction contains a specific item."""
        return any(item.sku == sku for item in self.items)

    def get_item_quantity(self, sku: str) -> int:
        """Get quantity of a specific item."""
        for item in self.items:
            if item.sku == sku:
                return item.quantity
        return 0


class POSDataLoader:
    """Loads and parses POS transaction data from JSON files."""

    def __init__(self, file_path: Optional[Path] = None):
        """
        Initialize POS data loader.

        Args:
            file_path: Path to JSON file, or None for default
        """
        self.file_path = file_path or DEFAULT_POS_PATH
        self._transactions: Optional[List[Transaction]] = None

    def load(self, file_path: Optional[Path] = None) -> List[Transaction]:
        """
        Load transactions from JSON file.

        Args:
            file_path: Path to JSON file, or use default

        Returns:
            List of Transaction objects
        """
        path = file_path or self.file_path

        if not path.exists():
            raise FileNotFoundError(f"POS data file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        self._transactions = self._parse_transactions(data)
        return self._transactions

    def _parse_transactions(self, data: Dict) -> List[Transaction]:
        """Parse raw JSON data into Transaction objects."""
        transactions = []

        for txn_data in data.get('transactions', []):
            # Parse items
            items = []
            for item_data in txn_data.get('items', []):
                item = TransactionItem(
                    sku=item_data['sku'],
                    name=item_data['name'],
                    quantity=item_data.get('quantity', 1),
                    price=item_data['price']
                )
                items.append(item)

            # Parse timestamp
            timestamp_str = txn_data.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()

            transaction = Transaction(
                transaction_id=txn_data.get('transaction_id', 'UNKNOWN'),
                timestamp=timestamp,
                items=items,
                total=txn_data.get('total', 0.0),
                payment_method=txn_data.get('payment_method', 'unknown')
            )
            transactions.append(transaction)

        return transactions

    @property
    def transactions(self) -> List[Transaction]:
        """Get loaded transactions, loading if necessary."""
        if self._transactions is None:
            self.load()
        return self._transactions

    def get_transactions_in_timeframe(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Transaction]:
        """
        Get transactions within a specific timeframe.

        Args:
            start_time: Start of timeframe
            end_time: End of timeframe

        Returns:
            List of transactions within timeframe
        """
        return [
            txn for txn in self.transactions
            if start_time <= txn.timestamp <= end_time
        ]

    def get_all_billed_items(self) -> Dict[str, int]:
        """
        Get all billed items across all transactions.

        Returns:
            Dictionary mapping SKU to total quantity
        """
        items = {}
        for txn in self.transactions:
            for item in txn.items:
                if item.sku in items:
                    items[item.sku] += item.quantity
                else:
                    items[item.sku] = item.quantity
        return items

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Transaction]:
        """Get a specific transaction by ID."""
        for txn in self.transactions:
            if txn.transaction_id == transaction_id:
                return txn
        return None

    def has_any_billing(self) -> bool:
        """Check if there are any transactions."""
        return len(self.transactions) > 0

    def get_total_billed(self) -> float:
        """Get total amount billed across all transactions."""
        return sum(txn.total for txn in self.transactions)
