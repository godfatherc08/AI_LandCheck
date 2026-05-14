"""
Veritas REST API - Open source document verification API
"""
import datetime
import os
import re
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse

from verification_engine.OCR.extraction import LandVerifyOCR
from verification_engine.agent.verifier import verify_document_offline

app = FastAPI(
    title="Veritas Document Verification API",
    description="AI-powered Nigerian land document verification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

print("Loading OCR engine...")
ocr_engine = LandVerifyOCR(enable_vision=True, sensitivity="medium")
print("API Ready!")


@app.get("/", tags=["Public"])
async def root():
    return {
        "name": "Veritas Document Verification API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "POST /verify": "Verify a document (returns JSON)",
            "POST /verify/report": "Verify and get markdown report",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Public"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


@app.post("/verify", tags=["Public"])
async def verify_document(file: UploadFile = File(...)):
    return await _run_verification(file)


@app.post("/verify/report", tags=["Public"], response_class=PlainTextResponse)
async def verify_document_report(file: UploadFile = File(...)):
    verification = await _run_verification(file)
    return verification["report"]


async def _run_verification(file: UploadFile) -> Dict:
    import time as timer

    file_ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    start_time = timer.time()

    try:
        result = ocr_engine.process_image_file(tmp_path)
        if not result:
            raise HTTPException(status_code=400, detail="Could not process image")

        doc_intel = result.get("document_intelligence", {})
        merged_fields = doc_intel.get("fields", {})
        all_warnings = doc_intel.get("warnings", [])
        stamp_detected = doc_intel.get("stamp_detected", False)
        signature_detected = doc_intel.get("signature_detected", False)
        vision_result = result.get("vision_forgery", {})
        full_text = result.get("page_text", "")

        flat_fields = {}
        for key, value in merged_fields.items():
            if isinstance(value, dict):
                flat_fields[key] = value.get("value") or value.get("cleaned_text")
            else:
                flat_fields[key] = value

        if not flat_fields.get("cof_number"):
            cof_match = re.search(r"LS/CO/\d{2}/\d{4}/\d{5}", full_text)
            if cof_match:
                flat_fields["cof_number"] = cof_match.group(0)

        image_np = np.array(result.get("image"))
        offline_verdict = verify_document_offline(
            image=image_np,
            ocr_results=[],
            extracted_fields=flat_fields,
            ocr_full_text=full_text
        )

        processing_time = timer.time() - start_time
        markdown_report = generate_markdown_report(
            filename=file.filename,
            processing_time=processing_time,
            flat_fields=flat_fields,
            vision_result=vision_result,
            offline_verdict=offline_verdict,
            all_warnings=all_warnings,
            stamp_detected=stamp_detected,
            signature_detected=signature_detected
        )

        return {
            "verification_id": f"VRT-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "overall_risk": offline_verdict.get("overall_risk", "MEDIUM"),
            "trust_score": offline_verdict.get("trust_score", 70),
            "squad_action": offline_verdict.get("squad_action", "HOLD_FUNDS_IN_ESCROW"),
            "processing_time_seconds": round(processing_time, 2),
            "report": markdown_report
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_markdown_report(filename: str, processing_time: float, flat_fields: Dict,
                             vision_result: Dict, offline_verdict: Dict, all_warnings: list,
                             stamp_detected: bool, signature_detected: bool) -> str:
    overall_risk = offline_verdict.get("overall_risk", "MEDIUM")
    trust_score = offline_verdict.get("trust_score", 70)
    squad_action = offline_verdict.get("squad_action", "HOLD_FUNDS_IN_ESCROW")
    recommendation = offline_verdict.get("recommendation", "Review required")

    if overall_risk == "LOW":
        risk_icon = "??"
    elif overall_risk == "MEDIUM":
        risk_icon = "??"
    elif overall_risk == "HIGH":
        risk_icon = "??"
    else:
        risk_icon = "??"

    report = f"""# VERIFICATION REPORT: {filename}

## EXECUTIVE SUMMARY

{risk_icon} **Verdict: {overall_risk} RISK**
**Trust Score: {trust_score}/100**
**Recommended Action: {squad_action}**
**Processing Time: {processing_time:.1f} seconds**

{recommendation}

## EXTRACTED INFORMATION

"""

    for field_name, field_value in flat_fields.items():
        if field_value and len(str(field_value)) < 100:
            report += f"- **{field_name.replace('_', ' ').title()}:** {field_value}\n"

    if not flat_fields:
        report += "- No fields were successfully extracted\n"

    report += f"""
## VERIFICATION CHECKS

| Check | Result |
|-------|--------|
| Official Stamp | {'? Detected' if stamp_detected else '? Not detected'} |
| Signature | {'? Detected' if signature_detected else '? Not detected'} |
| Vision Forensics | {vision_result.get('verdict', 'Not analyzed')} |

## FINDINGS

"""

    for warning in all_warnings[:5]:
        report += f"- ?? {warning}\n"

    if not all_warnings:
        report += "- No major issues detected\n"

    report += f"""
## RECOMMENDATIONS

### For the Buyer:
- {'?? Do NOT make any payment. This document shows signs of fraud.' if overall_risk in ['HIGH', 'CRITICAL'] else '? Payment can proceed with standard due diligence.'}

### For the Seller:
- {'Provide additional documentation to verify ownership.' if overall_risk in ['MEDIUM', 'HIGH', 'CRITICAL'] else 'Documentation appears complete.'}

### For the Bank:
- **Action: {squad_action}**
- {'Block payment immediately.' if squad_action == 'BLOCK_PAYMENT' else 'Hold funds in escrow pending manual review.' if squad_action == 'HOLD_FUNDS_IN_ESCROW' else 'Release payment.'}

## VERDICT

**Risk Level:** {overall_risk}  
**Trust Score:** {trust_score}/100  
**Confidence:** {'HIGH' if trust_score >= 80 else 'MEDIUM' if trust_score >= 60 else 'LOW'}

---
*Report generated by Veritas AI Document Verification System*
*Timestamp: {datetime.datetime.now().isoformat()}*
"""

    return report


if __name__ == "__main__":
    import uvicorn
    print("Starting Veritas API Server...")
    print("Available endpoints:")
    print("  POST /verify - JSON result")
    print("  POST /verify/report - Markdown report")
    print("  GET /docs - Interactive documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)
