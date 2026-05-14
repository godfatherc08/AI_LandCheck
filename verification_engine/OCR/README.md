# OCR Module Guide

The `verification_engine/OCR` directory is the document-ingestion and normalization boundary for the broader verification system. Its overarching function is to convert scanned land documents into a consistent intermediate representation that downstream modules can trust. Instead of directly producing a legal verdict, this layer focuses on extracting high-quality textual and spatial evidence that later verification logic can evaluate.

The runtime script `ocr_run.py` is the execution entrypoint. It initializes the OCR pipeline, loads a source file, processes each page, aggregates page-level results, and persists a consolidated output package for the rest of the engine. The generated output contains recognized text, token-level geometry, confidence information, page statistics, and optional forensic signals, making it suitable both for automated follow-up checks and for human inspection.

The core implementation in `extraction.py` handles PDF-to-image conversion, preprocessing, OCR inference, and result shaping. The design emphasizes robustness on mixed-quality scans by combining page rendering with image cleanup before recognition. This approach reduces noise before pattern extraction and helps preserve layout context, which is essential for later field mapping in land-document workflows.

Within the larger `verification_engine` architecture, this directory provides the evidentiary substrate on which every other module depends. If OCR quality degrades, field extraction and plausibility checks degrade as well, so this module is effectively the quality gate for the entire pipeline. Its implementation therefore prioritizes deterministic output structure, reproducible page handling, and storage of detailed metadata that can be audited when a verification outcome is challenged.

In practical terms, running this directory produces timestamped artifacts under `verification_results` so each verification session can be traced from source input to extracted evidence. That traceability enables threshold calibration, regression testing, and post-incident analysis without rerunning the original document unless needed.
