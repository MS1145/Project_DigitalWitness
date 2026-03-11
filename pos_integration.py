"""
pos_integration.py — Digital Witness POS Integration Module
============================================================
Handles mock POS transaction data, timestamp matching, and
operator-prompted verification for the Digital Witness system.

Usage in notebook:
    from pos_integration import (
        MockPOSDatabase, POSComparator,
        prompt_operator_verification, generate_pos_report
    )
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# MOCK POS DATABASE
# ============================================================

# Product catalogue — maps YOLO class groups to product names
# Used when generating mock transactions and displaying results
PRODUCT_CATALOGUE = {
    'grocery':    ['Milk (1L)',    'Bread',         'Orange Juice',
                   'Yoghurt',     'Butter',         'Eggs (6pk)'],
    'snacks':     ['Crisps',      'Chocolate Bar',  'Biscuits',
                   'Nuts (200g)', 'Granola Bar',    'Popcorn'],
    'beverages':  ['Water (500ml)','Soft Drink',    'Energy Drink',
                   'Coffee (can)','Iced Tea',       'Sports Drink'],
    'household':  ['Shampoo',     'Soap',           'Toothpaste',
                   'Deodorant',   'Face Wash',      'Hand Cream'],
    'frozen':     ['Ice Cream',   'Frozen Pizza',   'Frozen Veg',
                   'Fish Fingers','Frozen Chips',   'Ready Meal'],
}

ALL_PRODUCTS = [p for items in PRODUCT_CATALOGUE.values() for p in items]


def _random_transaction(session_id, timestamp, n_items_min=1, n_items_max=6):
    """Generate a single realistic mock transaction."""
    n_items = random.randint(n_items_min, n_items_max)
    items   = random.choices(ALL_PRODUCTS, k=n_items)

    # Build itemised receipt
    receipt = {}
    for item in items:
        receipt[item] = receipt.get(item, 0) + 1

    total_price = round(sum(
        random.uniform(0.50, 8.99) * qty
        for qty in receipt.values()
    ), 2)

    return {
        'session_id'   : session_id,
        'timestamp'    : timestamp.isoformat(),
        'items_billed' : n_items,
        'receipt'      : receipt,         # {product_name: quantity}
        'total_price'  : total_price,
        'payment_method': random.choice(['card', 'cash', 'contactless']),
        'cashier_id'   : f'C{random.randint(1, 5):02d}',
        'till_id'      : f'TILL-{random.randint(1, 4)}',
    }


class MockPOSDatabase:
    """
    Simulated POS database for testing.
    Generates realistic transactions with timestamps so the
    system can match a video session to a transaction by time.

    Usage:
        db = MockPOSDatabase()
        db.generate_sessions(n=20)
        db.save('pos_data/mock_transactions.json')

        # Find transaction for a video starting at a given time
        txn = db.find_by_timestamp(video_start_time, tolerance_seconds=60)
    """

    def __init__(self):
        self.transactions = []   # list of transaction dicts

    def generate_sessions(self, n=20, base_date=None):
        """
        Generate n mock shopping sessions spread across a day.

        Args:
            n         : number of transactions to generate
            base_date : datetime to anchor to (defaults to today)
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=9, minute=0,
                                                second=0, microsecond=0)

        self.transactions = []
        # Spread sessions across store opening hours (9am–9pm)
        for i in range(n):
            offset_minutes = random.randint(0, 720)   # 0–12 hours
            ts = base_date + timedelta(minutes=offset_minutes,
                                       seconds=random.randint(0, 59))
            txn = _random_transaction(
                session_id = f'SES-{i+1:04d}',
                timestamp  = ts,
                n_items_min= 1,
                n_items_max= 8,
            )
            self.transactions.append(txn)

        # Sort by timestamp
        self.transactions.sort(key=lambda x: x['timestamp'])
        print(f'Generated {n} mock transactions.')
        return self

    def add_suspicious_session(self, timestamp=None, items_billed=2,
                                items_detected=5):
        """
        Add a deliberately suspicious session for testing.
        The customer is billed for fewer items than detected.

        Args:
            items_billed   : what appears on the receipt
            items_detected : what YOLO would see them carrying
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Build a receipt with fewer items than detected
        receipt = {}
        items = random.choices(ALL_PRODUCTS, k=items_billed)
        for item in items:
            receipt[item] = receipt.get(item, 0) + 1

        txn = {
            'session_id'    : f'SES-SUSP-{random.randint(1000,9999)}',
            'timestamp'     : timestamp.isoformat(),
            'items_billed'  : items_billed,
            'receipt'       : receipt,
            'total_price'   : round(items_billed * random.uniform(1.5, 4.0), 2),
            'payment_method': random.choice(['card', 'cash', 'contactless']),
            'cashier_id'    : f'C{random.randint(1,5):02d}',
            'till_id'       : f'TILL-{random.randint(1,4)}',
            '_note'         : f'TEST: {items_detected} detected vs {items_billed} billed',
        }
        self.transactions.append(txn)
        print(f'Added suspicious session: {items_billed} billed, '
              f'{items_detected} detected (mismatch: {items_detected - items_billed})')
        return txn

    def find_by_timestamp(self, query_time, tolerance_seconds=120):
        """
        Find the closest transaction to a given timestamp.

        Args:
            query_time         : datetime or ISO string
            tolerance_seconds  : max allowed time difference

        Returns transaction dict or None if no match within tolerance.
        """
        if isinstance(query_time, str):
            query_time = datetime.fromisoformat(query_time)

        best_match = None
        best_delta = float('inf')

        for txn in self.transactions:
            txn_time = datetime.fromisoformat(txn['timestamp'])
            delta    = abs((query_time - txn_time).total_seconds())
            if delta < best_delta:
                best_delta  = delta
                best_match  = txn

        if best_delta <= tolerance_seconds:
            return best_match, best_delta
        return None, best_delta

    def save(self, path):
        """Save all transactions to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.transactions, f, indent=2)
        print(f'Saved {len(self.transactions)} transactions → {path}')

    def load(self, path):
        """Load transactions from a JSON file."""
        with open(path) as f:
            self.transactions = json.load(f)
        print(f'Loaded {len(self.transactions)} transactions from {path}')
        return self

    def summary(self):
        """Print a summary of the database."""
        if not self.transactions:
            print('No transactions loaded.')
            return
        print(f'Transactions  : {len(self.transactions)}')
        total_items = sum(t['items_billed'] for t in self.transactions)
        print(f'Total items   : {total_items}')
        print(f'Avg items/txn : {total_items / len(self.transactions):.1f}')
        times = [datetime.fromisoformat(t['timestamp'])
                 for t in self.transactions]
        print(f'Time range    : {min(times).strftime("%H:%M")} – '
              f'{max(times).strftime("%H:%M")}')


# ============================================================
# POS COMPARATOR
# ============================================================

class POSComparator:
    """
    Compares YOLO-detected product evidence against POS transaction data.

    The comparison logic:
      1. YOLO records what products the person was seen with
         (max_products_held, concealment events, checkout visited)
      2. POS provides what was billed
      3. Comparator flags mismatches and calculates a mismatch severity

    For thesis: this is your transactional integrity check —
    the second pillar of the Digital Witness system alongside
    behavioural analysis.
    """

    # Severity thresholds for item count mismatches
    MISMATCH_LOW      = 1   # 1 item unaccounted
    MISMATCH_MEDIUM   = 2   # 2 items unaccounted
    MISMATCH_HIGH     = 3   # 3+ items unaccounted

    def compare(self, person_summary, transaction):
        """
        Core comparison between video evidence and transaction.

        Args:
            person_summary : dict from PersonProductTracker.to_dict()
            transaction    : transaction dict from MockPOSDatabase

        Returns a detailed comparison result dict.
        """
        detected = person_summary['max_products_held']
        billed   = transaction['items_billed']
        diff     = detected - billed

        # Determine mismatch severity
        if diff <= 0:
            mismatch          = False
            mismatch_severity = 'NONE'
        elif diff == self.MISMATCH_LOW:
            mismatch          = True
            mismatch_severity = 'LOW'
        elif diff == self.MISMATCH_MEDIUM:
            mismatch          = True
            mismatch_severity = 'MEDIUM'
        else:
            mismatch          = True
            mismatch_severity = 'HIGH'

        # Build explanation for operator
        explanation_parts = []
        if mismatch:
            explanation_parts.append(
                f'{detected} product(s) detected by YOLO, '
                f'only {billed} billed at POS — '
                f'{diff} item(s) unaccounted for'
            )
        else:
            explanation_parts.append(
                f'{detected} product(s) detected, {billed} billed — counts match'
            )

        # Additional flags from tracker
        if person_summary['left_with_goods']:
            explanation_parts.append(
                'Person left store with goods without visiting checkout'
            )
        if person_summary['concealment_ratio'] > 0.3:
            explanation_parts.append(
                f'Backpack/bag present {person_summary["concealment_ratio"]:.0%} '
                f'of tracked time — concealment risk'
            )

        return {
            'session_id'       : transaction['session_id'],
            'products_detected': detected,
            'products_billed'  : billed,
            'difference'       : diff,
            'mismatch'         : mismatch,
            'mismatch_severity': mismatch_severity,
            'left_with_goods'  : person_summary['left_with_goods'],
            'concealment_ratio': person_summary['concealment_ratio'],
            'checkout_visited' : person_summary['checkout_visited'],
            'receipt'          : transaction.get('receipt', {}),
            'till_id'          : transaction.get('till_id', 'N/A'),
            'payment_method'   : transaction.get('payment_method', 'N/A'),
            'explanation'      : ' | '.join(explanation_parts),
            'pos_integrated'   : True,
        }

    def compare_no_transaction(self, person_summary):
        """Fallback when no matching POS transaction found."""
        detected = person_summary['max_products_held']
        flags    = []
        if person_summary['left_with_goods']:
            flags.append('Left store with goods — no checkout visit recorded')
        if person_summary['concealment_ratio'] > 0.3:
            flags.append(f'Concealment risk: bag present '
                         f'{person_summary["concealment_ratio"]:.0%} of time')

        return {
            'session_id'       : None,
            'products_detected': detected,
            'products_billed'  : None,
            'difference'       : None,
            'mismatch'         : True if flags else None,
            'mismatch_severity': 'UNKNOWN',
            'left_with_goods'  : person_summary['left_with_goods'],
            'concealment_ratio': person_summary['concealment_ratio'],
            'checkout_visited' : person_summary['checkout_visited'],
            'receipt'          : {},
            'explanation'      : ('No matching POS transaction found. '
                                   + ' | '.join(flags) if flags
                                   else 'No matching POS transaction found.'),
            'pos_integrated'   : False,
        }


# ============================================================
# OPERATOR PROMPT — interactive verification
# ============================================================

def prompt_operator_verification(inference_result, pos_db=None,
                                  video_timestamp=None, tolerance_seconds=120):
    """
    Interactive operator verification prompt.

    Shows the operator:
      1. What YOLO detected (products, persons, behaviour)
      2. The matched POS transaction (if found by timestamp)
      3. Asks operator to confirm or override the billed count

    This simulates the real retail security workflow where a human
    operator reviews the AI's findings before any action is taken.

    Args:
        inference_result  : dict returned by run_inference()
        pos_db            : MockPOSDatabase instance (None = manual entry only)
        video_timestamp   : datetime of when the video was recorded
                            (used to find matching transaction)
        tolerance_seconds : how close the timestamps need to be

    Returns:
        list of POSComparator result dicts, one per tracked person
    """
    comparator = POSComparator()
    results    = []

    sep   = '=' * 65
    sep2  = '-' * 65

    print(sep)
    print('  DIGITAL WITNESS — OPERATOR VERIFICATION')
    print(sep)
    print(f'  Video duration  : {inference_result["duration"]:.1f}s')
    print(f'  Persons tracked : {len(inference_result["person_summaries"])}')
    print(f'  Classification  : {inference_result["overall_class"].upper()} '
          f'({inference_result["overall_conf"]:.1%} confidence)')
    print()

    # ---- Try to find matching POS transaction by timestamp ----
    matched_transaction = None
    match_delta         = None

    if pos_db is not None and video_timestamp is not None:
        matched_transaction, match_delta = pos_db.find_by_timestamp(
            video_timestamp, tolerance_seconds
        )
        if matched_transaction:
            print(f'  POS Match Found : {matched_transaction["session_id"]} '
                  f'(Δ{match_delta:.0f}s)')
            print(f'  Till            : {matched_transaction["till_id"]}')
            print(f'  Items billed    : {matched_transaction["items_billed"]}')
            print(f'  Total           : £{matched_transaction["total_price"]:.2f}')
            print(f'  Payment         : {matched_transaction["payment_method"]}')
            print()
            print('  Receipt:')
            for item, qty in matched_transaction['receipt'].items():
                print(f'    {qty}x  {item}')
        else:
            print(f'  POS Match       : NONE (closest was Δ{match_delta:.0f}s '
                  f'— outside {tolerance_seconds}s tolerance)')
        print()

    # ---- Per-person verification ----
    for tid, person in inference_result['person_summaries'].items():
        print(sep2)
        print(f'  PERSON {tid} — Video Evidence')
        print(sep2)

        # Show what YOLO recorded
        print(f'  Max products held   : {person["max_products_held"]}')
        print(f'  Concealment ratio   : {person["concealment_ratio"]:.0%} '
              f'(backpack/bag detected)')
        print(f'  Checkout visited    : {person["checkout_visited"]}')
        print(f'  Left with goods     : {person["left_with_goods"]}')
        print(f'  Frames tracked      : {person["total_frames"]}')
        print()

        # ---- Operator input ----
        if matched_transaction:
            # Pre-fill from POS match, operator can override
            print(f'  POS transaction shows {matched_transaction["items_billed"]} '
                  f'item(s) billed.')
            try:
                override = input(
                    f'  Confirm billed count [{matched_transaction["items_billed"]}] '
                    f'or enter correct value: '
                ).strip()
                if override == '':
                    confirmed_billed = matched_transaction['items_billed']
                else:
                    confirmed_billed = int(override)
                # Build a working transaction with operator-confirmed count
                working_txn = dict(matched_transaction)
                working_txn['items_billed'] = confirmed_billed
            except (ValueError, EOFError):
                # Non-interactive environment (e.g. automated test)
                confirmed_billed = matched_transaction['items_billed']
                working_txn      = matched_transaction
                print(f'  [Auto] Using POS value: {confirmed_billed}')
        else:
            # No POS match — operator must enter manually
            try:
                manual = input(
                    f'  No POS match found. Enter billed item count for Person {tid}: '
                ).strip()
                confirmed_billed = int(manual) if manual else 0
            except (ValueError, EOFError):
                confirmed_billed = 0
                print(f'  [Auto] Defaulting to 0 billed items')

            # Build a minimal transaction from manual entry
            working_txn = {
                'session_id'    : f'MANUAL-{tid}',
                'timestamp'     : (video_timestamp.isoformat()
                                   if video_timestamp else datetime.now().isoformat()),
                'items_billed'  : confirmed_billed,
                'receipt'       : {'(manually entered)': confirmed_billed},
                'total_price'   : 0.0,
                'payment_method': 'unknown',
                'till_id'       : 'N/A',
                'cashier_id'    : 'N/A',
            }

        # ---- Run comparison ----
        result = comparator.compare(person, working_txn)
        results.append(result)

        # ---- Show result ----
        print()
        status = '✓  MATCH' if not result['mismatch'] else f'✗  MISMATCH [{result["mismatch_severity"]}]'
        print(f'  Result : {status}')
        print(f'  Detected: {result["products_detected"]}  |  '
              f'Billed: {result["products_billed"]}  |  '
              f'Difference: {result["difference"]}')
        print(f'  Detail : {result["explanation"]}')
        print()

    print(sep)
    total_mismatches = sum(1 for r in results if r['mismatch'])
    if total_mismatches:
        print(f'  ⚠  {total_mismatches} MISMATCH(ES) DETECTED — Escalate for human review')
    else:
        print(f'  ✓  All transactions verified — No mismatches')
    print(f'  NOTE: This is an advisory system. All decisions require human review.')
    print(sep)

    return results


# ============================================================
# POS REPORT GENERATOR
# ============================================================

def generate_pos_report(inference_result, pos_comparison_results,
                         case_id, output_dir):
    """
    Generate a JSON POS integrity report and a human-readable summary.
    Saved alongside the case file for audit trail.

    Args:
        inference_result       : dict from run_inference()
        pos_comparison_results : list from prompt_operator_verification()
        case_id                : string case ID
        output_dir             : Path to outputs directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build report ----
    mismatches = [r for r in pos_comparison_results if r['mismatch']]

    report = {
        'case_id'          : case_id,
        'generated_at'     : datetime.now().isoformat(),
        'video_summary'    : {
            'duration'       : inference_result['duration'],
            'persons_tracked': len(inference_result['person_summaries']),
            'classification' : inference_result['overall_class'],
            'confidence'     : inference_result['overall_conf'],
        },
        'pos_comparisons'  : pos_comparison_results,
        'total_persons'    : len(pos_comparison_results),
        'mismatches_found' : len(mismatches),
        'integrity_status' : 'FAIL' if mismatches else 'PASS',
        'mismatch_details' : [
            {
                'session_id'       : r['session_id'],
                'detected'         : r['products_detected'],
                'billed'           : r['products_billed'],
                'unaccounted'      : r['difference'],
                'severity'         : r['mismatch_severity'],
                'left_with_goods'  : r['left_with_goods'],
                'concealment_ratio': r['concealment_ratio'],
            }
            for r in mismatches
        ],
        'advisory_note': (
            'This report is advisory only. Item counts from computer vision '
            'are estimates. All mismatches require human verification before '
            'any action is taken.'
        ),
    }

    # Save JSON report
    report_path = output_dir / f'{case_id}_pos_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ---- Print summary ----
    print(f'\nPOS Integrity Report — {case_id}')
    print(f'  Status          : {report["integrity_status"]}')
    print(f'  Persons checked : {report["total_persons"]}')
    print(f'  Mismatches      : {report["mismatches_found"]}')
    if mismatches:
        for m in report['mismatch_details']:
            print(f'  → Session {m["session_id"]}: '
                  f'{m["detected"]} detected, {m["billed"]} billed, '
                  f'{m["unaccounted"]} unaccounted [{m["severity"]}]')
    print(f'  Report saved    : {report_path}')

    return report, str(report_path)
