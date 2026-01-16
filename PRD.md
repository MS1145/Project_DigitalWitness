# Product Requirements Document (PRD)
## Digital Witness
Bias-Aware, Explainable Retail Security Assistant

Version: 0.1 (MVP / IPD Prototype)  
Author: [Your Name]  
Date: 2026-01-16  
Status: In Progress

---

## 1. Product Overview

### 1.1 Problem Statement

Current AI-powered retail security systems operate as opaque black boxes. They are unable to reliably distinguish between intentional shoplifting and accidental or vulnerable behaviour such as children misunderstanding purchases or elderly customers making mistakes. This results in false accusations, loss of customer trust, ethical concerns, and legal liability.

Additionally, existing systems often treat behavioural evidence and POS transactions as separate silos, making it difficult to determine whether items physically taken by a customer were actually billed.

---

## 2. Product Vision

Digital Witness is designed as a **Blameless AI Assistant** that supports — not replaces — human decision-making in retail security.

The system analyses a customer’s **physical interaction with products** and correlates it with **POS transaction data** to determine whether all products taken by the customer have been billed.

The system does not determine guilt. Instead, it provides:
- An intent assessment
- Short digital forensic video clips highlighting suspicious moments
- A clear explanation of why an alert was generated

Final accountability always lies with a **human operator**.

---

## 3. Core Capabilities

### Behavioural Analysis
- Detect product pickup and interaction events
- Identify suspicious actions such as concealment or bypassing checkout
- Extract short video clips of these events

### Transactional Verification
- Ingest POS transaction data
- Cross-check detected product interactions against billed items
- Identify discrepancies between physical actions and billing records

### Intent Assessment
- Fuse behavioural evidence with POS discrepancies
- Generate an intent score indicating likelihood of shoplifting versus error
- Categorise alerts by severity

### Explainability & Digital Forensics
- Provide video-based forensic evidence
- Generate human-readable explanations
- Preserve evidence for later review

### Bias-Aware Handling
- No facial recognition or identity tracking
- Special consideration for vulnerable scenarios such as children or elderly individuals
- Alerts are advisory and require human validation

---

## 4. MVP Scope (IPD Prototype)

The MVP demonstrates feasibility rather than production-level accuracy.

### Included
- Upload and process a recorded video
- Pose-based behavioural analysis
- Mock POS data ingestion
- Product interaction vs billing cross-check
- Intent score generation
- Alert triggering
- Explanation text and forensic clips
- Case output saved for audit

### Excluded
- Live retail deployment
- Facial recognition
- Fully trained deep learning models
- Bias mitigation strategies
- Edge deployment optimisation

---

## 5. Future Roadmap

### Version 1.0
- Sequence-based learning models (CNN-LSTM / GRU)
- Automated product counting improvements
- SHAP-based POS explanations
- Operator dashboard

### Version 2.0
- Bias mitigation strategies
- Differentiated response logic
- Edge deployment
- Performance benchmarking
- Exportable forensic reports

---

## 6. Success Criteria (MVP)

- End-to-end pipeline runs successfully
- Product interactions are detected
- POS data is cross-checked
- Alerts are generated for discrepancies
- Explanations and clips are displayed
- Human validation is required for final decisions

---

## 7. Constraints & Limitations

- Uses staged videos only
- Uses mocked POS data
- Small dataset
- Academic prototype only

---
