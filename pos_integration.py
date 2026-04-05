import json, random
from datetime import datetime, timedelta
from pathlib import Path


PRODUCTS = [
    {"sku": "ITEM001", "name": "Snack Bar",     "price": 2.99},
    {"sku": "ITEM002", "name": "Soda Bottle",   "price": 1.99},
    {"sku": "ITEM003", "name": "Chocolate Box", "price": 5.99},
    {"sku": "ITEM004", "name": "Energy Drink",  "price": 3.49},
    {"sku": "ITEM005", "name": "Chips Bag",     "price": 2.49},
]


class MockPOSDatabase:
    """Simulates POS transaction data for audit comparison against video evidence."""

    def __init__(self):
        self.sessions = []

    def generate_sessions(self, n=20):
        random.seed(42)
        now = datetime.now()
        for i in range(n):
            ts = now - timedelta(minutes=random.randint(0, 480))
            items = random.sample(PRODUCTS, k=random.randint(1, 3))
            total = sum(it["price"] for it in items)
            self.sessions.append({
                "session_id": f"TXN-{i+1:04d}",
                "timestamp": ts.isoformat(),
                "items_billed": len(items),
                "items": [{"sku": it["sku"], "name": it["name"], "price": it["price"]} for it in items],
                "total": round(total, 2),
                "flagged": False,
            })

    def add_suspicious_session(self, timestamp, items_billed, items_detected):
        # items_detected > items_billed = potential theft
        self.sessions.append({
            "session_id": f"TXN-SUSP-001",
            "timestamp": timestamp.isoformat(),
            "items_billed": items_billed,
            "items_detected_on_camera": items_detected,
            "discrepancy": items_detected - items_billed,
            "flagged": True,
            "flag_reason": f"{items_detected} items detected by camera but only {items_billed} scanned at POS",
        })

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"sessions": self.sessions, "generated_at": datetime.now().isoformat()}, f, indent=2)
        print(f"[INFO] POS data saved to {path}")

    def summary(self):
        flagged = [s for s in self.sessions if s.get("flagged")]
        print(f"[INFO] POS sessions: {len(self.sessions)} total, {len(flagged)} flagged")
        for s in flagged:
            print(f"  {s['session_id']}: {s.get('flag_reason', 'flagged')}")
