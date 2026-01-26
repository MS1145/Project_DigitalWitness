# Product Requirements Document (PRD)

## Digital Witness

Bias-Aware, Explainable Retail Security Assistant


## 1. Product Overview

### 1.1 Problem Statement

Current AI-powered retail security systems operate as opaque black boxes. They are unable to reliably distinguish between intentional shoplifting and accidental or vulnerable behaviour such as children misunderstanding purchases or elderly customers making mistakes. This results in false accusations, erosion of customer trust, ethical concerns, and potential legal liability.

Additionally, existing systems often treat behavioural evidence and POS transactions as separate silos, making it difficult to determine whether items physically taken by a customer were actually billed.

---

## 2. Product Vision

Digital Witness is designed as a **Blameless AI Assistant** that supports — not replaces — human decision-making in retail security.

The system analyses a customer’s **physical interaction with products** and correlates it with **POS transaction data** to determine whether all products taken by the customer have been billed.

The system does **not** determine guilt or confirm shoplifting. Instead, it provides:

- An intent risk assessment
- An Intent Risk Level (Level 1–5) indicating severity and confidence
- Short digital forensic video clips highlighting suspicious moments
- Clear, human-readable explanations supporting review

All alerts are **advisory**. Final accountability and decisions — including whether to assist a customer, dismiss an alert, or escalate to store management — always remain with a **human operator**.

---

## 3. Design Principle: Use of Existing AI Models

Digital Witness intentionally builds upon **established open-source AI models** rather than developing low-level computer vision systems from scratch.

This approach ensures:

- Technical reliability and reproducibility
- Alignment with industry and academic best practices
- Focus on higher-level research contributions

Examples include:

- Pre-trained pose estimation models for anonymised behavioural analysis
- Open-source object or interaction detection models where applicable
- Established explainability libraries for transparent reasoning

The project’s **original contribution** lies in:

- Behavioural and transactional data fusion
- Intent risk inference across full customer journeys
- Explainable digital forensic evidence generation
- Bias-aware, human-in-the-loop decision support design

---

## 4. Core Capabilities

### Behavioural Analysis

- Detect product pickup and interaction events using anonymised pose data
- Identify suspicious actions such as concealment or bypassing checkout
- Extract short video clips of relevant behavioural moments

### Transactional Verification

- Ingest POS transaction data (mocked for MVP)
- Cross-check detected product interactions against billed items
- Identify discrepancies between physical actions and billing records

### Intent Risk Assessment

- Fuse behavioural indicators with POS discrepancies
- Generate an intent risk score and risk level (1–5)
- Support differentiated response recommendations

### Explainability & Digital Forensics

- Provide video-based forensic evidence
- Generate human-readable explanations
- Preserve case evidence for audit and review

### Bias-Aware & Fair Handling

- No facial recognition or identity inference
- Special consideration for vulnerable scenarios (e.g., children, elderly customers)
- Alerts are advisory and require human validation

---

## 5. MVP Scope (IPD Prototype)

The MVP demonstrates **end-to-end feasibility**, not production-level accuracy.

### Included

- Upload and process pre-recorded video
- Pose-based behavioural analysis
- Mock POS data ingestion
- Product interaction versus billing cross-check
- Intent risk score and alert level generation
- Explanation text and forensic clips
- Case output saved for audit

### Excluded

- Live retail deployment
- Facial recognition or personal identification
- Fully trained deep learning pipelines
- Bias mitigation strategies (measurement only)
- Edge deployment and optimisation

---

## 6. Future Roadmap

### Version 1.0 (Post-IPD)

- Sequence-based learning models (e.g., GRU / CNN-LSTM)
- Improved product interaction and counting logic
- SHAP-based explanations for transactional data
- Operator-facing dashboard

### Version 2.0 (Final Product)

- Bias mitigation strategies
- Differentiated response logic for vulnerable groups
- Low-cost edge deployment
- Performance benchmarking
- Exportable forensic and accountability reports

---

## 7. Success Criteria (MVP)

The MVP is considered successful if:

- A video can be uploaded and processed end-to-end
- Product interactions are detected
- POS data is cross-checked against physical actions
- Alerts are generated based on risk levels
- Explanations and forensic clips are available
- Human validation is required for all final decisions

---

## 8. Constraints & Limitations

- Uses staged videos only
- Uses mocked POS data
- Limited dataset size
- Academic prototype, not a production system

---
