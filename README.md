# Digital Witness

Digital Witness is a bias-aware, explainable AI prototype designed to assist retail security staff by analysing customer behaviour and POS transaction data to detect potential shoplifting events.

This repository contains the **IPD MVP prototype**, demonstrating end-to-end feasibility.

---

## Project Goals

- Demonstrate behavioural analysis using anonymised pose data
- Cross-check physical product interactions with POS billing data
- Generate explainable alerts supported by forensic video evidence
- Keep humans accountable for all final decisions

---

## Repository Structure

- `PRD.md` – Product Requirements Document
- `src/` – Core source code modules
- `data/` – Sample videos and mock POS data
- `notebooks/` – Experimental notebooks
- `outputs/` – Generated case files and results

---

## How to Run (Prototype)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the prototype notebook or script:
   ```bash
   python src/main.py
   ```

3. Review generated intent scores, alerts, and explanations.

---

## Notes

- This is an academic prototype, not a production system.
- No personal or identifying data is processed.
- Accuracy and bias mitigation are addressed in later project phases.

---
