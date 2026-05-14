"""
VERITAS - Complete Land Verification Pipeline
Supports: PDF, JPG, JPEG, PNG, BMP, TIFF
"""

from verification_engine.OCR.extraction import LandVerifyOCR
from verification_engine.agent.verifier import verify_document_offline
from pathlib import Path
import time
import json
import numpy as np
from datetime import datetime
from PIL import Image
import re

print("=" * 70)
print("VERITAS - LAND VERIFICATION ENGINE")
print("=" * 70)
print("🔍 Document Intelligence: ENABLED (zone-aware OCR)")
print("🔧 Offline Verification Tools: ENABLED (5 forensic checks)")
print("=" * 70)

                       
ocr_engine = LandVerifyOCR(enable_vision=True, sensitivity='medium')

                                       
input_file = "test.png"                 
file_ext = Path(input_file).suffix.lower()

                         
output_dir = Path("verification_results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = Path(input_file).stem
start_time = time.time()

                                              
                                                 
                                              
merged_fields = {}
all_warnings = []
stamp_detected = False
signature_detected = False
vision_result = {}
first_page_image = None
full_text = ""
page_count = 0

                                              
                            
                                              

if file_ext == '.pdf':
    print(f"\n📄 Processing PDF: {input_file}")
    print("-" * 70)

                              
    page_results = ocr_engine.process_pdf_pages(input_file, max_pages=3)

    if not page_results:
        print("❌ No pages extracted from PDF")
        exit(1)

    page_count = len(page_results)

                                  
    for page in page_results:
        doc_intel = page.get('document_intelligence', {})

                      
        for field_name, field_data in doc_intel.get('fields', {}).items():
            if field_name not in merged_fields:
                merged_fields[field_name] = field_data
            elif field_data.get('confidence', 0) > merged_fields[field_name].get('confidence', 0):
                merged_fields[field_name] = field_data

                          
        all_warnings.extend(doc_intel.get('warnings', []))

                                   
        if doc_intel.get('stamp_detected'):
            stamp_detected = True
        if doc_intel.get('signature_detected'):
            signature_detected = True

                         
    first_page = page_results[0]
    first_page_image = first_page.get('image')
    vision_result = first_page.get('vision_forgery', {})

                                 
    full_text = '\n\n'.join([p.get('page_text', '') for p in page_results])

elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
    print(f"\n📸 Processing Image: {input_file}")
    print("-" * 70)

                          
    result = ocr_engine.process_image_file(input_file)

    if not result:
        print("❌ Failed to process image")
        exit(1)

    page_count = 1

                  
    doc_intel = result.get('document_intelligence', {})
    merged_fields = doc_intel.get('fields', {})
    all_warnings = doc_intel.get('warnings', [])
    stamp_detected = doc_intel.get('stamp_detected', False)
    signature_detected = doc_intel.get('signature_detected', False)
    vision_result = result.get('vision_forgery', {})
    first_page_image = result.get('image')
    full_text = result.get('page_text', '')

                     
    print(f"\n📊 Overall OCR confidence: {doc_intel.get('overall_ocr_confidence', 0):.1%}")
    print(f"🕹️ STAMP DETECTION: {'✅' if stamp_detected else '❌'}")
    print(f"✍️ SIGNATURE DETECTION: {'✅' if signature_detected else '❌'}")

    print(f"\n📋 EXTRACTED FIELDS ({len(merged_fields)} found):")
    if merged_fields:
        for field_name, field_data in list(merged_fields.items())[:10]:
                                                
            if isinstance(field_data, dict):
                value = field_data.get('value') or field_data.get('cleaned_text', '')
                confidence = field_data.get('confidence', 0)
            else:
                value = str(field_data)
                confidence = 0
            if value and len(str(value)) < 100:                          
                print(f"   • {field_name}: '{value}' (conf: {confidence:.1%})")

                                     
        for field_name, field_data in merged_fields.items():
            if isinstance(field_data, dict):
                value = field_data.get('value') or field_data.get('cleaned_text', '')
            else:
                value = str(field_data)
            if value and len(str(value)) > 100:
                print(f"   • {field_name}: '{str(value)[:80]}...'")
    else:
        print("   No fields extracted - check OCR quality")

    if all_warnings:
        print(f"\n⚠️ VALIDATION WARNINGS ({len(all_warnings)}):")
        for warning in all_warnings[:3]:
            print(f"   • {warning}")

                            
    if vision_result:
        print("\n" + "=" * 70)
        print("VISION FORENSIC ANALYSIS")
        print("=" * 70)
        if vision_result.get('looks_forged'):
            print(f"⚠️ VERDICT: {vision_result.get('verdict', 'UNKNOWN')}")
        else:
            print(f"✅ VERDICT: {vision_result.get('verdict', 'AUTHENTIC')}")
            print(f"   {vision_result.get('summary', 'No forgery indicators detected')}")

else:
    print(f"❌ Unsupported file type: {file_ext}")
    print("Supported: .pdf, .jpg, .jpeg, .png, .bmp, .tiff")
    exit(1)

                                              
                                                     
                                              

                                                                   
flat_fields = {}
for field_name, field_data in merged_fields.items():
    if isinstance(field_data, dict):
                                  
        value = field_data.get('value') or field_data.get('cleaned_text') or field_data.get('raw_text')
        if value:
            flat_fields[field_name] = value
    else:
        flat_fields[field_name] = field_data

                                                           
if not flat_fields.get('cof_number'):
    cof_match = re.search(r'LS/CO/\d{2}/[A-Z]{2}/\d{6}', full_text)
    if cof_match:
        flat_fields['cof_number'] = cof_match.group(0)

if not flat_fields.get('term_years'):
    if '99 YEARS' in full_text.upper() or 'NINETY-NINE' in full_text.upper():
        flat_fields['term_years'] = '99'

if not flat_fields.get('plot_number'):
    plot_match = re.search(r'PLOT\s*NO[.:]?\s*(\d+)', full_text, re.IGNORECASE)
    if plot_match:
        flat_fields['plot_number'] = plot_match.group(1)

print(f"\n📋 FLATTENED FIELDS for verification: {len(flat_fields)} fields")
for k, v in flat_fields.items():
    if v and len(str(v)) < 80:
        print(f"   • {k}: {v}")

                                              
                      
                                              

print("\n" + "=" * 70)
print("OFFLINE DOCUMENT VERIFICATION")
print("=" * 70)

if first_page_image is None:
    print("❌ No image available for offline verification")
    offline_verdict = {
        "overall_risk": "MEDIUM",
        "squad_action": "HOLD_FUNDS_IN_ESCROW",
        "trust_score": 50,
        "total_suspicion_weight": 0.5,
        "recommendation": "No image data available for verification"
    }
else:
    image_np = np.array(first_page_image)

                                                    
    offline_verdict = verify_document_offline(
        image=image_np,
        ocr_results=[],
        extracted_fields=flat_fields,                        
        ocr_full_text=full_text
    )

                 
print(f"\n📊 VERIFICATION RESULTS:")
print(f"   Risk Level: {offline_verdict['overall_risk']}")
print(f"   Squad Action: {offline_verdict['squad_action']}")
print(f"   Trust Score: {offline_verdict['trust_score']}/100")

print(f"\n💡 RECOMMENDATION: {offline_verdict['recommendation']}")

                                              
                    
                                              

                                              
                                    
                                              

                                         
verification_report = {
    "verification_id": f"VRT-{timestamp}",
    "timestamp": datetime.now().isoformat(),
    "source_file": input_file,
    "file_type": file_ext,
    "processing_time_seconds": round(time.time() - start_time, 2),

                                                  
                          
                                                  
    "executive_summary": {
        "overall_risk": offline_verdict['overall_risk'],
        "squad_action": offline_verdict['squad_action'],
        "trust_score": offline_verdict['trust_score'],
        "one_line_verdict": offline_verdict['recommendation'],
        "confidence_level": "HIGH" if offline_verdict['trust_score'] >= 80 else "MEDIUM" if offline_verdict[
                                                                                                'trust_score'] >= 60 else "LOW",
        "requires_manual_review": offline_verdict['overall_risk'] in ["MEDIUM", "HIGH", "CRITICAL"]
    },

                                                  
                                      
                                                  
    "document_intelligence": {
        "ocr_confidence_percent": round(doc_intel.get('overall_ocr_confidence', 0) * 100, 1),
        "zones_detected": doc_intel.get('zones_found', []),
        "stamp": {
            "detected": stamp_detected,
            "confidence_percent": round(doc_intel.get('stamp_confidence', 0) * 100, 1) if stamp_detected else 0,
            "has_circular_shape": doc_intel.get('seal_detected', False),
            "has_stamp_ink": doc_intel.get('has_stamp_ink', False)
        },
        "signature": {
            "detected": signature_detected,
            "confidence_percent": round(doc_intel.get('signature_confidence', 0) * 100, 1) if signature_detected else 0,
            "expected_count": 3,
            "found_count": 1 if signature_detected else 0
        },
        "document_orientation_corrected_degrees": doc_intel.get('document_orientation', 0),
        "validation_warnings": all_warnings[:10],
        "validation_passed_count": len([w for w in all_warnings if "No" not in w]),
        "validation_failed_count": len([w for w in all_warnings if "No" in w or "not" in w.lower()])
    },

                                                  
                         
                                                  
    "extracted_fields": {},
    "extraction_summary": {
        "total_fields_extracted": len(flat_fields),
        "critical_fields_present": [],
        "critical_fields_missing": []
    }
}

                                           
critical_fields = ["cof_number", "property_owner", "issue_date", "term_years"]
for field_name, field_value in flat_fields.items():
                                                
    confidence = 0
    if field_name in merged_fields:
        if isinstance(merged_fields[field_name], dict):
            confidence = merged_fields[field_name].get('confidence', 0)
    else:
        confidence = 0.85                                  

    verification_report["extracted_fields"][field_name] = {
        "value": field_value,
        "confidence_percent": round(confidence * 100, 1),
        "status": "VALID" if confidence > 0.7 else "LOW_CONFIDENCE" if confidence > 0.4 else "UNRELIABLE"
    }

    if field_name in critical_fields:
        if field_value and confidence > 0.5:
            verification_report["extraction_summary"]["critical_fields_present"].append(field_name)
        else:
            verification_report["extraction_summary"]["critical_fields_missing"].append(field_name)

                                              
                             
                                              

vision = vision_result if vision_result else {}
vision_checks = vision.get('checks', {})

verification_report["vision_forensics"] = {
    "overall_verdict": vision.get('verdict', 'NOT_ANALYZED'),
    "looks_forged": vision.get('looks_forged', False),
    "summary": vision.get('summary', 'No analysis performed'),
    "sensitivity_setting": vision.get('sensitivity', 'medium'),
    "checks_performed": len(vision_checks),
    "checks_failed": len([c for c in vision_checks.values() if c.get('looks_wrong')]),
    "checks_passed": len([c for c in vision_checks.values() if not c.get('looks_wrong')]),
    "detailed_checks": {}
}

                            
for check_name, check_data in vision_checks.items():
    verification_report["vision_forensics"]["detailed_checks"][check_name] = {
        "verdict": "FAIL" if check_data.get('looks_wrong') else "PASS",
        "confidence_percent": round(check_data.get('confidence', 0) * 100, 1),
        "reason": check_data.get('reason', ''),
        "details": check_data.get('details', {})
    }

                            
if vision.get('anomaly_regions'):
    verification_report["vision_forensics"]["anomaly_regions"] = vision['anomaly_regions']

                                              
                                      
                                              

offline_results = offline_verdict.get('tool_results', {})
verification_report["offline_verification"] = {
    "overall_risk": offline_verdict['overall_risk'],
    "total_suspicion_weight": offline_verdict['total_suspicion_weight'],
    "tools_executed": len(offline_results),
    "tools_failed": len([t for t in offline_results.values() if t.get('severity') not in ['PASS', 'LOW']]),
    "tool_details": {}
}

                             
for tool_name, tool_result in offline_results.items():
    if tool_result:
        verification_report["offline_verification"]["tool_details"][tool_name] = {
            "severity": tool_result.get('severity', 'UNKNOWN'),
            "weight": tool_result.get('weight', 0),
            "passed": tool_result.get('severity') == 'PASS',
            "explanation": tool_result.get('explanation', ''),
            "issues": tool_result.get('issues', [])[:5]                     
        }

                                              
                                            
                                              

                                                     
recommendations = []
if not stamp_detected:
    recommendations.append("Official government stamp not detected - verify document authenticity")
if not signature_detected:
    recommendations.append("Required signatures missing - document may be incomplete or fraudulent")
if flat_fields.get('cof_number') == 'LANDSREGISTRY':
    recommendations.append("C of O number appears invalid - expected format LS/CO/XX/XXXX/XXXXX")
if len(flat_fields) < 3:
    recommendations.append("Limited fields extracted - document may be incomplete")
if offline_verdict['overall_risk'] in ['HIGH', 'CRITICAL']:
    recommendations.append("Document failed critical verification checks - do not proceed with transaction")

verification_report["verdict"] = {
    "risk_level": offline_verdict['overall_risk'],
    "risk_score": offline_verdict['trust_score'],
    "squad_api_action": offline_verdict['squad_action'],
    "human_readable_verdict": offline_verdict['recommendation'],
    "recommended_actions": recommendations if recommendations else ["Document verified. Proceed with transaction."],
    "for_buyer": "Do not make payment" if offline_verdict['overall_risk'] in ['HIGH',
                                                                              'CRITICAL'] else "Payment can proceed" if
    offline_verdict['overall_risk'] == 'LOW' else "Hold payment pending review",
    "for_seller": "Provide additional documentation" if offline_verdict['overall_risk'] in ['MEDIUM', 'HIGH',
                                                                                            'CRITICAL'] else "Documentation complete",
    "for_bank": offline_verdict['squad_action']
}
from verification_engine.agent import generate_verification_report

human_report = generate_verification_report(
    extracted_fields=flat_fields,
    vision_forensics=vision_result if vision_result else {},
    offline_verdict=offline_verdict,
    document_intelligence=doc_intel,
    validation_warnings=all_warnings,
    ocr_confidence=doc_intel.get('overall_ocr_confidence', 0.95),
    processing_time=time.time() - start_time,
    document_name=Path(input_file).name
)

# Save report as Markdown file
report_path = output_dir / f"{base_name}_report_{timestamp}.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(human_report)

print(f"📄 Human-readable report saved to: {report_path}")
print(f"\n📄 Human-readable report saved to: {report_path}")

               
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print(f"✅ Time: {verification_report['processing_time_seconds']:.2f} seconds")
print(f"📄 Pages: {page_count}")
print(f"📋 Fields extracted: {len(flat_fields)}")
print(f"🕹️ Stamp detected: {stamp_detected}")
print(f"✍️ Signature detected: {signature_detected}")

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)
print(f"   Risk Level: {offline_verdict['overall_risk']}")
print(f"   Trust Score: {offline_verdict['trust_score']}/100")
print(f"   Squad Action: {offline_verdict['squad_action']}")
print(f"\n   {offline_verdict['recommendation']}")
if recommendations:
    print("\n   Recommended Actions:")
    for rec in recommendations[:3]:
        print(f"   • {rec}")
print("=" * 70)

print(f"\n💾 Detailed report saved to: {output_dir}")
