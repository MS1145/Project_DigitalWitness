"""
POS (Point of Sale) module for Digital Witness.
Handles transaction data loading and real-time POS simulation.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
from enum import Enum

from .config import DEFAULT_POS_PATH, POS_PRODUCT_CATALOG


@dataclass
class TransactionItem:
    """A single item in a transaction."""
    sku: str
    name: str
    quantity: int
    price: float


@dataclass
class Transaction:
    """A POS transaction."""
    transaction_id: str
    timestamp: datetime
    items: List[TransactionItem]
    total: float
    payment_method: str

    def get_item_skus(self) -> List[str]:
        skus = []
        for item in self.items:
            skus.extend([item.sku] * item.quantity)
        return skus

    def has_item(self, sku: str) -> bool:
        return any(item.sku == sku for item in self.items)


class POSDataLoader:
    """Loads POS transaction data from JSON files."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or DEFAULT_POS_PATH
        self._transactions: Optional[List[Transaction]] = None

    def load(self, file_path: Optional[Path] = None) -> List[Transaction]:
        path = file_path or self.file_path
        if not path.exists():
            raise FileNotFoundError(f"POS data file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        self._transactions = self._parse_transactions(data)
        return self._transactions

    def _parse_transactions(self, data: Dict) -> List[Transaction]:
        transactions = []
        for txn_data in data.get('transactions', []):
            items = [
                TransactionItem(
                    sku=item['sku'], name=item['name'],
                    quantity=item.get('quantity', 1), price=item['price']
                )
                for item in txn_data.get('items', [])
            ]
            try:
                timestamp = datetime.fromisoformat(txn_data.get('timestamp', ''))
            except ValueError:
                timestamp = datetime.now()
            transactions.append(Transaction(
                transaction_id=txn_data.get('transaction_id', 'UNKNOWN'),
                timestamp=timestamp, items=items,
                total=txn_data.get('total', 0.0),
                payment_method=txn_data.get('payment_method', 'unknown')
            ))
        return transactions

    @property
    def transactions(self) -> List[Transaction]:
        if self._transactions is None:
            self.load()
        return self._transactions

    def get_all_billed_items(self) -> Dict[str, int]:
        items = {}
        for txn in self.transactions:
            for item in txn.items:
                items[item.sku] = items.get(item.sku, 0) + item.quantity
        return items


class SessionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    VOIDED = "voided"


@dataclass
class Product:
    sku: str
    name: str
    price: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Product":
        return cls(sku=data["sku"], name=data["name"], price=data["price"])


@dataclass
class ScanEvent:
    sku: str
    name: str
    price: float
    quantity: int
    timestamp: datetime
    terminal_id: str = "POS001"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id, "sku": self.sku, "name": self.name,
            "price": self.price, "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(), "terminal_id": self.terminal_id
        }


@dataclass
class POSSession:
    session_id: str
    terminal_id: str
    started_at: datetime
    items: List[ScanEvent] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    completed_at: Optional[datetime] = None

    def add_item(self, event: ScanEvent) -> None:
        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot add items to {self.status.value} session")
        self.items.append(event)

    def get_total(self) -> float:
        return sum(item.price * item.quantity for item in self.items)

    def complete(self) -> Transaction:
        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot complete {self.status.value} session")
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
        transaction_items = [
            TransactionItem(sku=item.sku, name=item.name, quantity=item.quantity, price=item.price)
            for item in self.items
        ]
        return Transaction(
            transaction_id=self.session_id, timestamp=self.completed_at,
            items=transaction_items, total=self.get_total(), payment_method="simulated"
        )

    def void(self) -> None:
        self.status = SessionStatus.VOIDED
        self.completed_at = datetime.now()


class POSSimulator:
    """Interactive POS simulator for real-time mode."""

    def __init__(self, catalog: Optional[List[Dict]] = None):
        catalog_data = catalog or POS_PRODUCT_CATALOG
        self.catalog: Dict[str, Product] = {
            item["sku"]: Product.from_dict(item) for item in catalog_data
        }
        self._current_session: Optional[POSSession] = None
        self._transaction_history: List[Transaction] = []
        self._on_scan_callbacks: List[Callable[[ScanEvent], None]] = []
        self._on_complete_callbacks: List[Callable[[Transaction], None]] = []

    def get_catalog(self) -> List[Product]:
        return list(self.catalog.values())

    def get_product(self, sku: str) -> Optional[Product]:
        return self.catalog.get(sku)

    def start_session(self, terminal_id: str = "POS001") -> POSSession:
        if self._current_session and self._current_session.status == SessionStatus.ACTIVE:
            self.void_session()
        session_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._current_session = POSSession(
            session_id=session_id, terminal_id=terminal_id, started_at=datetime.now()
        )
        return self._current_session

    def scan_item(self, sku: str, quantity: int = 1) -> Optional[ScanEvent]:
        if self._current_session is None or self._current_session.status != SessionStatus.ACTIVE:
            self.start_session()
        product = self.get_product(sku)
        if product is None:
            return None
        event = ScanEvent(
            sku=product.sku, name=product.name, price=product.price,
            quantity=quantity, timestamp=datetime.now(),
            terminal_id=self._current_session.terminal_id
        )
        self._current_session.add_item(event)
        for callback in self._on_scan_callbacks:
            callback(event)
        return event

    def complete_transaction(self) -> Optional[Transaction]:
        if self._current_session is None or self._current_session.status != SessionStatus.ACTIVE:
            return None
        transaction = self._current_session.complete()
        self._transaction_history.append(transaction)
        for callback in self._on_complete_callbacks:
            callback(transaction)
        self._current_session = None
        return transaction

    def void_session(self) -> bool:
        if self._current_session is None or self._current_session.status != SessionStatus.ACTIVE:
            return False
        self._current_session.void()
        self._current_session = None
        return True

    @property
    def current_session(self) -> Optional[POSSession]:
        return self._current_session

    @property
    def has_active_session(self) -> bool:
        return self._current_session is not None and self._current_session.status == SessionStatus.ACTIVE

    @property
    def transaction_history(self) -> List[Transaction]:
        return self._transaction_history.copy()

    def on_scan(self, callback: Callable[[ScanEvent], None]) -> None:
        self._on_scan_callbacks.append(callback)

    def on_complete(self, callback: Callable[[Transaction], None]) -> None:
        self._on_complete_callbacks.append(callback)

    def get_transactions_for_analysis(self) -> List[Transaction]:
        transactions = self._transaction_history.copy()
        if self.has_active_session and self._current_session.items:
            temp_items = [
                TransactionItem(sku=item.sku, name=item.name, quantity=item.quantity, price=item.price)
                for item in self._current_session.items
            ]
            transactions.append(Transaction(
                transaction_id=f"{self._current_session.session_id}_PENDING",
                timestamp=datetime.now(), items=temp_items,
                total=self._current_session.get_total(), payment_method="pending"
            ))
        return transactions
