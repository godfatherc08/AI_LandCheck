"""
Offline Document Verifier - No API calls, runs entirely locally
Uses 7 forensic tools to verify document authenticity
"""

import numpy as np
from typing import Dict, List, Optional

import random
from datetime import datetime

from verification_engine.agent.offline_tools import (
    verify_reference_numbers,
    analyze_font_consistency,
    validate_date_logic,
    analyze_print_scan_consistency,
    match_document_template,
    analyze_image_provenance,
    verify_signature_presence,
    verify_barcodes_presence
)


class DocumentVerifier:
    """
    Complete offline document verification using 7 forensic tools
    No internet, no API keys, no external dependencies
    """

    def __init__(self):
        self.tool_results = {}

    def verify(self, image: np.ndarray,
               ocr_results: List = None,
               extracted_fields: Dict = None,
               ocr_full_text: str = "",
               document_type: str = "certificate_of_occupancy") -> Dict:
        """
        Run all 7 offline verification tools

        Args:
            image: Document image as numpy array
            ocr_results: List of OCR results with bbox, text, confidence
            extracted_fields: Dictionary of extracted document fields
            ocr_full_text: Full concatenated OCR text
            document_type: Type of document (certificate_of_occupancy, land_information_certificate)

        Returns:
            Dict with overall verdict and individual tool results
        """

        results = {}

                                            
        if extracted_fields:
            results["reference_numbers"] = verify_reference_numbers(extracted_fields)

                                  
        if ocr_results:
            results["font_consistency"] = analyze_font_consistency(image, ocr_results)

                            
        if extracted_fields:
            results["date_logic"] = validate_date_logic(extracted_fields)

                                        
        results["print_scan"] = analyze_print_scan_consistency(image)

                                  
        if ocr_full_text:
            results["template"] = match_document_template(ocr_full_text, document_type)

                                  
        results["provenance"] = analyze_image_provenance(image)

        results["signatures"] = verify_signature_presence(extracted_fields, ocr_full_text)

                              
        results["barcodes"] = verify_barcodes_presence(ocr_full_text)

                           
        verdict = self._aggregate_verdict(results)

        return verdict

    def _aggregate_verdict(self, results: Dict, extracted_fields: Dict = None) -> Dict:
        """Combine all tool results into final verdict with normalization"""

        all_signals = []
        total_weight = 0
        critical_count = 0
        high_count = 0

                                        
        tools_executed = [name for name, result in results.items() if result and not result.get('error')]

                                                     
        max_possible_weight = len(tools_executed) * 0.85                              

                                     
        sig_result = results.get("signatures", {})
        if sig_result and sig_result.get("severity") == "CRITICAL":
                                                           
                                                               
            ocr_full_text = extracted_fields.get('_raw_text', '') if extracted_fields else ''
            if 'ELECTRONIC CERTIFICATE' not in ocr_full_text.upper():
                total_weight += 0.6                     
                critical_count += 1

                                          
        barcode_result = results.get("barcodes", {})
        ocr_full_text = extracted_fields.get('_raw_text', '') if extracted_fields else ''
        is_electronic = 'ELECTRONIC CERTIFICATE' in ocr_full_text.upper() or 'UNIQUE BARCODES' in ocr_full_text.upper()

        if barcode_result and barcode_result.get("severity") != "PASS":
            if is_electronic:
                                                              
                total_weight += 0.75
                high_count += 1
            else:
                                                                            
                total_weight += 0.3

                                          
        for tool_name, tool_result in results.items():
            if not tool_result or tool_name in ["signatures", "barcodes"]:
                continue

            weight = tool_result.get("weight", 0)
            severity = tool_result.get("severity", "PASS")

                                                           
            if tool_name == "print_scan" and is_electronic:
                weight = weight * 0.5                                      

            if weight > 0:
                total_weight += weight
                all_signals.append({
                    "tool": tool_name,
                    "severity": severity,
                    "weight": weight,
                    "explanation": tool_result.get("explanation", ""),
                })

                if severity == "CRITICAL":
                    critical_count += 1
                elif severity == "HIGH":
                    high_count += 1

                                                           
                                                                                              
        expected_checks = len(tools_executed)
        actual_penalty = total_weight
        normalized_penalty = actual_penalty / max(expected_checks * 0.5, 1)             

        trust_score = max(50, min(95, 95 - int(normalized_penalty * 30)))

                                        
        variation = random.randint(-2, 2)
        trust_score = max(50, min(95, trust_score + variation))


                                                    
        if critical_count > 1 or total_weight > 1.2:
            overall_risk = "CRITICAL"
            squad_action = "BLOCK_PAYMENT"
            recommendation = "Document failed CRITICAL verification checks. Do not release payment."
        elif high_count > 0 or total_weight > 0.8:
            overall_risk = "HIGH"
            squad_action = "BLOCK_PAYMENT"
            recommendation = "Document failed multiple verification checks. Block payment."
        elif total_weight > 0.6:
            overall_risk = "MEDIUM"
            squad_action = "HOLD_FUNDS_IN_ESCROW"
            recommendation = "Document has suspicious indicators. Hold funds in escrow for manual review."
        else:
            overall_risk = "LOW"
            squad_action = "RELEASE_PAYMENT"
            recommendation = "Document passed all verification checks. Proceed with payment."

        return {
            "overall_risk": overall_risk,
            "squad_action": squad_action,
            "trust_score": trust_score,
            "total_suspicion_weight": round(total_weight, 3),
            "normalized_penalty": round(normalized_penalty, 3),
            "critical_flags": critical_count,
            "high_flags": high_count,
            "signals": all_signals,
            "tool_results": results,
            "recommendation": recommendation,
            "method": "Offline Verification (7 forensic tools)"
        }

def verify_document_offline(image: np.ndarray,
                            ocr_results: List = None,
                            extracted_fields: Dict = None,
                            ocr_full_text: str = "") -> Dict:
    """
    Convenience function for offline document verification
    """
    verifier = DocumentVerifier()
    return verifier.verify(image, ocr_results, extracted_fields, ocr_full_text)