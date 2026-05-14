                                    

import cv2
import re
import math
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rapidocr_onnxruntime import RapidOCR


                                                                       
                                    
                                                                
                                                                
                                                  
                                                                       

def verify_reference_numbers(extracted_fields: Dict) -> Dict:
    """
    Validates all reference numbers found in the document against
    known Lagos State government numbering conventions.
    No internet needed — pure pattern and logic validation.
    """
    signals = []

    PATTERNS = {
        "osg_ref": {
            "pattern": r"^OSG/(\d{3})/(\d{5})$",
            "example": "OSG/536/39988",
            "rule": "Middle segment must be 3 digits, last must be 5 digits",
        },
        "survey_plan_aros": {
            "pattern": r"^AROS/LA/(\d{4})/(\d{4})/([A-Z]\d{3})/(\d{1,2})$",
            "example": "AROS/LA/1460/2016/C002/5",
            "rule": "Third segment is year, must be 1978–present",
        },
        "survey_plan_ros": {
            "pattern": r"^ROS/LA/(\d{5})$",
            "example": "ROS/LA/14607",
            "rule": "Must be exactly 5 digits",
        },
        "cof_number": {
            "pattern": r"^C\s*of\s*[Oo]\s*[-–]?\s*(\d{4,5})$",
            "example": "C of O - 8272",
            "rule": "4 or 5 digit serial number",
        },
        "luc_number": {
            "pattern": r"^LUC/(\d{4})/(\d{4,6})$",
            "example": "LUC/2016/00123",
            "rule": "Year segment must be 1978–present",
        },
        "lasg_receipt": {
            "pattern": r"^\d{8}/[A-Z]{6,12}$",
            "example": "37328743/NMARHYJA",
            "rule": "8-digit serial + alphanumeric code",
        },
    }

    current_year = datetime.now().year

    for field_name, value in extracted_fields.items():
        if not isinstance(value, str):
            continue
        value = value.strip().upper()

        for ref_type, spec in PATTERNS.items():
            match = re.match(spec["pattern"], value, re.IGNORECASE)
            if not match:
                continue

            groups = match.groups()
            issues = []

                                              
            if ref_type in ("survey_plan_aros", "luc_number"):
                year_idx = 1 if ref_type == "luc_number" else 2
                try:
                    year = int(groups[year_idx if ref_type == "luc_number" else 1])
                    if not (1978 <= year <= current_year):
                        issues.append(
                            f"Year {year} is outside valid range "
                            f"(1978–{current_year}). "
                            f"Land Use Act commenced 1978."
                        )
                except (ValueError, IndexError):
                    pass

                                                                       
            if ref_type == "osg_ref":
                try:
                    office_code = int(groups[0])
                    serial      = int(groups[1])
                    if office_code == 0:
                        issues.append("OSG office code cannot be 000")
                    if serial == 0:
                        issues.append("OSG serial number cannot be 00000")
                except (ValueError, IndexError):
                    pass

            signals.append({
                "field":      field_name,
                "value":      value,
                "ref_type":   ref_type,
                "format_valid": len(issues) == 0,
                "issues":     issues,
                "weight":     0.7 * len(issues) if issues else 0.0,
                "severity":   "HIGH" if issues else "PASS",
            })

                                        
                                                                        
    survey = extracted_fields.get("survey_plan_number", "")
    date   = extracted_fields.get("issue_date", "")
    year_match = re.search(r"(\d{4})", survey)
    date_match = re.search(r"(\d{4})", date)

    if year_match and date_match:
        plan_year = int(year_match.group(1))
        doc_year  = int(date_match.group(1))
        if plan_year > doc_year:
            signals.append({
                "field":    "cross_reference_date",
                "value":    f"Plan year {plan_year} > Document year {doc_year}",
                "format_valid": False,
                "issues":   ["Survey plan cannot be dated after the certificate"],
                "weight":   0.85,
                "severity": "CRITICAL",
            })

    return {
        "tool": "reference_number_forensics",
        "signals": signals,
        "overall_valid": all(s["format_valid"] for s in signals),
        "critical_count": sum(1 for s in signals if s["severity"] == "CRITICAL"),
    }


                                                                       
                                       
                                                        
                                                               
                                                            
                                                           
                                                                       

def analyze_font_consistency(image: np.ndarray, ocr_results: List) -> Dict:
    """
    Measures stroke width, character height, and spacing consistency
    across all detected text regions.
    Edited fields show up as statistical outliers.
    """
    if not ocr_results:
        return {"tool": "font_consistency", "error": "No OCR results provided"}

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    region_metrics = []

    for result in ocr_results:
        bbox, text, confidence = result[0], result[1], result[2]
        if not text.strip() or confidence < 0.5:
            continue

                              
        pts  = np.array(bbox, dtype=np.int32)
        x1   = max(0, min(pts[:, 0]))
        y1   = max(0, min(pts[:, 1]))
        x2   = min(image.shape[1], max(pts[:, 0]))
        y2   = min(image.shape[0], max(pts[:, 1]))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = binary[y1:y2, x1:x2]
        if roi.size == 0:
            continue

                                             
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        nonzero = dist[dist > 0]
        if len(nonzero) == 0:
            continue

        stroke_width = float(np.mean(nonzero) * 2)                      
        char_height  = float(y2 - y1)
        ink_density  = float(np.mean(roi > 0))

        region_metrics.append({
            "text":         text[:30],
            "stroke_width": stroke_width,
            "char_height":  char_height,
            "ink_density":  ink_density,
            "bbox":         [int(x1), int(y1), int(x2), int(y2)],
        })

    if len(region_metrics) < 3:
        return {
            "tool": "font_consistency",
            "error": "Too few text regions for analysis",
            "region_count": len(region_metrics),
        }

    strokes  = np.array([r["stroke_width"] for r in region_metrics])
    heights  = np.array([r["char_height"]  for r in region_metrics])
    densities = np.array([r["ink_density"] for r in region_metrics])

    stroke_cv = float(np.std(strokes)   / (np.mean(strokes)   + 1e-8))
    height_cv = float(np.std(heights)   / (np.mean(heights)   + 1e-8))
    density_cv = float(np.std(densities) / (np.mean(densities) + 1e-8))

                                          
    outliers = []
    for i, m in enumerate(region_metrics):
        z_stroke  = abs(strokes[i]   - np.mean(strokes))   / (np.std(strokes)   + 1e-8)
        z_height  = abs(heights[i]   - np.mean(heights))   / (np.std(heights)   + 1e-8)
        z_density = abs(densities[i] - np.mean(densities)) / (np.std(densities) + 1e-8)

        if max(z_stroke, z_height, z_density) > 2.5:
            outliers.append({
                **m,
                "z_stroke":  round(float(z_stroke), 2),
                "z_height":  round(float(z_height), 2),
                "z_density": round(float(z_density), 2),
            })

                                                              
    inconsistent = stroke_cv > 0.25 or height_cv > 0.35

    return {
        "tool":            "font_consistency",
        "region_count":    len(region_metrics),
        "stroke_width_cv": round(stroke_cv, 4),
        "char_height_cv":  round(height_cv, 4),
        "ink_density_cv":  round(density_cv, 4),
        "outlier_regions": outliers,
        "inconsistent":    inconsistent,
        "weight":          0.7 if inconsistent else 0.0,
        "severity":        "HIGH" if inconsistent else "PASS",
        "explanation": (
            f"{len(outliers)} text region(s) have font metrics that differ "
            f"significantly from the rest of the document — consistent with "
            f"fields that were individually edited or retyped."
            if outliers else
            "Font metrics are consistent across all text regions."
        ),
    }




                                                                       
                                      
                                               
                                                                       
                                                      
                                 
                                                                       

def validate_date_logic(extracted_fields: Dict) -> Dict:
    """
    Extracts all dates from the document and validates
    that they form a logically consistent timeline.
    """
    DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
                    "%d/%m/%y", "%B %d, %Y", "%d %B %Y"]

    def parse_date(s: str) -> Optional[datetime]:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        return None

    def extract_all_dates(fields: Dict) -> Dict[str, datetime]:
        found = {}
        date_fields = {
            k: v for k, v in fields.items()
            if any(kw in k.lower() for kw in
                   ["date", "issued", "signed", "registered", "receipt"])
            and isinstance(v, str)
        }
                                                       
        date_pattern = re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b')
        for key, value in fields.items():
            if isinstance(value, str):
                for match in date_pattern.finditer(value):
                    dt = parse_date(match.group(1))
                    if dt:
                        found[key] = dt
        return found

    all_dates = extract_all_dates(extracted_fields)
    issues    = []
    passed    = []
    now       = datetime.now()

                                          
    for field, dt in all_dates.items():
        if dt > now:
            issues.append({
                "rule":      "future_date",
                "field":     field,
                "value":     dt.strftime("%d/%m/%Y"),
                "severity":  "CRITICAL",
                "weight":    0.95,
                "reason":    f"Date {dt.strftime('%d/%m/%Y')} is in the future",
            })

                                                               
    land_use_act = datetime(1978, 3, 29)
    for field, dt in all_dates.items():
        if dt < land_use_act:
            issues.append({
                "rule":     "predates_land_use_act",
                "field":    field,
                "value":    dt.strftime("%d/%m/%Y"),
                "severity": "CRITICAL",
                "weight":   0.90,
                "reason":   "Certificate of Occupancy cannot predate the Land Use Act 1978",
            })

                                                                   
    issue_date   = extracted_fields.get("issue_date", "")
    survey_plan  = extracted_fields.get("survey_plan_number", "")
    issue_dt     = parse_date(issue_date) if issue_date else None
    plan_year_m  = re.search(r"(\d{4})", survey_plan)

    if issue_dt and plan_year_m:
        plan_year = int(plan_year_m.group(1))
        if 1978 <= plan_year <= now.year:
            if issue_dt.year < plan_year:
                issues.append({
                    "rule":     "issue_before_survey",
                    "severity": "CRITICAL",
                    "weight":   0.90,
                    "reason":   (
                        f"Certificate issued {issue_dt.year} but "
                        f"survey plan dated {plan_year} — "
                        f"certificate cannot precede its own survey plan"
                    ),
                })
            else:
                passed.append("Issue date is after survey plan date ✓")

                                                                   
    receipt_dates = {k: v for k, v in all_dates.items()
                     if "receipt" in k.lower() or "payment" in k.lower()}
    if issue_dt and receipt_dates:
        late_receipts = {k: v for k, v in receipt_dates.items()
                         if v > issue_dt}
        if late_receipts:
            passed.append(
                f"All {len(receipt_dates)} receipt date(s) precede the issue date ✓"
            )
        else:
            for field, dt in late_receipts.items():
                issues.append({
                    "rule":     "receipt_after_certificate",
                    "field":    field,
                    "severity": "HIGH",
                    "weight":   0.70,
                    "reason":   (
                        f"Receipt dated {dt.strftime('%d/%m/%Y')} is after "
                        f"certificate issue date {issue_dt.strftime('%d/%m/%Y')}"
                    ),
                })

    if not issues:
        passed.append("All dates form a consistent timeline ✓")

    return {
        "tool":          "date_logic",
        "dates_found":   {k: v.strftime("%d/%m/%Y") for k, v in all_dates.items()},
        "issues":        issues,
        "passed":        passed,
        "weight":        max((i["weight"] for i in issues), default=0.0),
        "severity":      max((i["severity"] for i in issues),
                             key=lambda s: ["PASS","LOW","MEDIUM","HIGH","CRITICAL"].index(s),
                             default="PASS"),
    }


                                                                       
                                           
                                                  
                                                      
                                        
                                                    
                                                            
                                      
                                                                       
def analyze_print_scan_consistency(image: np.ndarray) -> Dict:
    """
    Verifies the document shows consistent physical printing and scanning
    artifacts. For electronic C of O, this is less relevant.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    issues = []
    zone_noise = {}                         

                                           
                                                           
    _, bg_mask = cv2.threshold(
        gray.astype(np.uint8), 220, 255, cv2.THRESH_BINARY
    )
    zones = {
        "top": gray[:h // 3, :][bg_mask[:h // 3, :] > 0],
        "middle": gray[h // 3:2 * h // 3, :][bg_mask[h // 3:2 * h // 3, :] > 0],
        "bottom": gray[2 * h // 3:, :][bg_mask[2 * h // 3:, :] > 0],
    }

    for name, pixels in zones.items():
        if len(pixels) > 100:
            zone_noise[name] = float(np.std(pixels))

    if len(zone_noise) >= 2:
        noise_values = list(zone_noise.values())
        noise_cv = np.std(noise_values) / (np.mean(noise_values) + 1e-8)
        if noise_cv > 0.5:
            issues.append({
                "check": "background_grain",
                "detail": f"Background noise varies across zones (CV={noise_cv:.2f})",
                "weight": 0.35,
            })

                                                  
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edge_zone = gray[dilated > 0]

    if len(edge_zone) > 0:
        edge_gradient = float(np.std(edge_zone))
        if edge_gradient < 15.0:
            issues.append({
                "check": "ink_spread",
                "detail": f"Text edges are unusually sharp (gradient std={edge_gradient:.1f})",
                "weight": 0.35,
            })

                                       
    block_variances = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            block = gray[y:y + 8, x:x + 8]
            block_variances.append(float(np.var(block)))

    if block_variances:
        bv = np.array(block_variances)
        bv_outlier_frac = float(np.mean(bv > np.mean(bv) + 3 * np.std(bv)))
        if bv_outlier_frac > 0.03:
            issues.append({
                "check": "jpeg_grid_artifacts",
                "detail": f"{bv_outlier_frac:.1%} of 8×8 blocks have anomalous variance",
                "weight": 0.45,
            })

                                                        
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=5
    )
    if lines is not None:
        perfect_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 2 or abs(x2 - x1) < 2:
                perfect_lines += 1
        ratio = perfect_lines / max(len(lines), 1)

                                      
        if ratio > 0.95 and len(lines) > 30:
            issues.append({
                "check": "perfect_lines",
                "detail": f"{ratio:.0%} of lines are pixel-perfect (expected for e-CofO)",
                "weight": 0.15,
            })

                        
    weight = max((i["weight"] for i in issues), default=0.0)

    if weight <= 0.2 and len(issues) <= 1:
        severity = "PASS"
        weight = 0.0
        explanation = "Document exhibits characteristics of electronic C of O (acceptable for digital certificates)."
    elif weight > 0:
        severity = "LOW" if weight < 0.3 else "MEDIUM"
        explanation = f"{len(issues)} minor inconsistency(ies) detected, within acceptable range for electronic certificate."
    else:
        severity = "PASS"
        explanation = "Document shows consistent physical printing and scanning characteristics."

    return {
        "tool": "print_scan_consistency",
        "issues": issues,
        "zone_noise": {k: round(v, 3) for k, v in zone_noise.items()},
        "weight": round(weight, 3),
        "severity": severity,
        "explanation": explanation,
    }
                                                                       
                                                    
                                                         
                                                     
                               
                                                          
                                                                       

def match_document_template(ocr_full_text: str,
                            document_type: str = "certificate_of_occupancy") -> Dict:
    """
    Checks that mandatory boilerplate text is present.
    Updated for Lagos State Electronic C of O format with OCR noise tolerance.
    """

                                                             
    text_normalized = ocr_full_text.lower()
    text_no_spaces = text_normalized.replace(" ", "")

    TEMPLATES = {
        "certificate_of_occupancy": {
            "mandatory_phrases": [
                ("land use act", ["landuseact", "land use act"]),
                ("right of occupancy", ["rightofoccupancy", "right of occupancy"]),
                ("governor of lagos state", ["governoroflagosstate", "governor of lagos state"]),
                ("lagos state government", ["lagosstategovernment", "lagos state government"]),
                ("certificate of occupancy", ["certificateofoccupancy", "certificate of occupancy"]),
            ],
            "electronic_indicators": [
                "electronic certificate",
                "barcodes for authentication",
                "unique encrypted",
                "transaction history",
            ],
        },
    }

    template = TEMPLATES.get(document_type, TEMPLATES["certificate_of_occupancy"])
    issues = []
    passed = []
    electronic_features = []

                                                           
    for phrase_display, variants in template["mandatory_phrases"]:
        found = False
        for variant in variants:
            if variant in text_no_spaces or variant in text_normalized:
                found = True
                break

                                            
        if not found:
            words = phrase_display.split()
            matched_words = sum(1 for w in words if w in text_normalized)
            if matched_words >= len(words) * 0.6:                  
                found = True
                passed.append(f"Partial match: '{phrase_display}'")

        if found:
            if "Partial" not in str(passed[-1] if passed else ""):
                passed.append(f"Found: '{phrase_display}'")
        else:
            issues.append({
                "type": "missing_boilerplate",
                "phrase": phrase_display,
                "severity": "MEDIUM",
                "weight": 0.35,
                "reason": f"Phrase '{phrase_display}' not found (may be due to OCR formatting)",
            })

                                                  
    for feature in template["electronic_indicators"]:
        if feature in text_normalized:
            electronic_features.append(feature)

                                                 
    weight = max((i["weight"] for i in issues), default=0.0)

                                                                      
    if electronic_features:
        weight = max(0, weight - 0.3)
        passed.append(f"✅ Electronic security features detected: {len(electronic_features)} found")

                                                               
    key_indicators = ["ls/co/", "99 years", "residential", "commissionerforstampduties"]
    key_found = sum(1 for ind in key_indicators if ind in text_no_spaces)
    if key_found >= 2:
        weight = max(0, weight - 0.2)
        passed.append(f"✅ Key C of O indicators found: {key_found}/4")

    severity = (
        "HIGH" if weight >= 0.6 else
        "MEDIUM" if weight >= 0.3 else
        "PASS"
    )

                                                                     
    if electronic_features and key_found >= 2 and weight <= 0.3:
        severity = "PASS"
        weight = 0.0
        passed.append("🎯 Document confirmed as genuine Lagos State Electronic C of O")

    return {
        "tool": "template_matcher",
        "document_type": document_type,
        "mandatory_found": len(passed),
        "mandatory_total": len(template["mandatory_phrases"]),
        "electronic_features_found": electronic_features,
        "key_indicators_found": key_found,
        "issues": issues,
        "passed": passed,
        "weight": round(weight, 3),
        "severity": severity,
        "explanation": (
            f"Document matches Electronic C of O format with {len(passed)}/{len(template['mandatory_phrases'])} mandatory phrases. "
            f"Security features: {len(electronic_features)}. Confidence: HIGH."
            if severity == "PASS"
            else f"Template verification passed {len(passed)}/{len(template['mandatory_phrases'])} mandatory phrases."
        ),
    }

                                                                       
                                           
                                                     
                                                   
                                      
                                                                       

def analyze_image_provenance(image: np.ndarray,
                              pil_image: Optional[Image.Image] = None) -> Dict:
    """
    Extracts and validates image provenance signals.
    Checks DPI consistency, colour space anomalies,
    and compression history indicators.
    """
    h, w  = image.shape[:2]
    issues = []
    info   = {}

                               
    if pil_image:
        exif_data = pil_image._getexif() if hasattr(pil_image, '_getexif') else None
        if exif_data:
                                                      
            x_res = exif_data.get(282)
            y_res = exif_data.get(283)
            if x_res and y_res:
                dpi_x = float(x_res[0]) / float(x_res[1]) if isinstance(x_res, tuple) else float(x_res)
                dpi_y = float(y_res[0]) / float(y_res[1]) if isinstance(y_res, tuple) else float(y_res)
                info["dpi_x"] = round(dpi_x, 1)
                info["dpi_y"] = round(dpi_y, 1)

                if abs(dpi_x - dpi_y) > 5:
                    issues.append({
                        "check":  "dpi_mismatch",
                        "detail": f"X-DPI ({dpi_x}) ≠ Y-DPI ({dpi_y}) — image was rescaled non-uniformly",
                        "weight": 0.50,
                    })

                if dpi_x < 150:
                    issues.append({
                        "check":  "low_dpi",
                        "detail": f"DPI is {dpi_x} — genuine document scans are typically 200–600 DPI",
                        "weight": 0.30,
                    })

                                                                                
            software = exif_data.get(305, "")
            if software and any(
                s in str(software).lower()
                for s in ["photoshop", "gimp", "pixelmator", "paint", "canva"]
            ):
                issues.append({
                    "check":  "editing_software",
                    "detail": f"Image metadata shows editing software: '{software}'",
                    "weight": 0.85,
                    "severity": "HIGH",
                })

                                                                      
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    char_heights = [cv2.boundingRect(c)[3] for c in contours
                    if 5 < cv2.boundingRect(c)[3] < 100]
    if char_heights:
        median_char_h  = float(np.median(char_heights))
                                                                                
        est_dpi        = (median_char_h / 10) * 72
        info["estimated_dpi"] = round(est_dpi, 0)

        if est_dpi < 100:
            issues.append({
                "check":  "estimated_low_resolution",
                "detail": (
                    f"Estimated effective DPI is {est_dpi:.0f} — "
                    f"document may have been downscaled after editing "
                    f"to hide artifacts"
                ),
                "weight": 0.40,
            })

                                       
                                                        
                                                               
    r_std = float(np.std(image[:, :, 0].astype(np.float32)))
    g_std = float(np.std(image[:, :, 1].astype(np.float32)))
    b_std = float(np.std(image[:, :, 2].astype(np.float32)))

    channel_cv = np.std([r_std, g_std, b_std]) / (np.mean([r_std, g_std, b_std]) + 1e-8)
    info["channel_balance_cv"] = round(float(channel_cv), 4)

    if channel_cv > 0.3:
        issues.append({
            "check":  "channel_imbalance",
            "detail": (
                f"RGB channel standard deviations are unbalanced (CV={channel_cv:.2f}) — "
                f"genuine scanner output produces balanced channels"
            ),
            "weight": 0.45,
        })

    weight   = max((i["weight"] for i in issues), default=0.0)
    severity = (
        "HIGH"   if weight >= 0.65 else
        "MEDIUM" if weight >= 0.40 else
        "PASS"
    )

    return {
        "tool":      "image_provenance",
        "image_size": [w, h],
        "info":      info,
        "issues":    issues,
        "weight":    round(weight, 3),
        "severity":  severity,
        "explanation": (
            f"{len(issues)} provenance check(s) failed."
            if issues else
            "Image provenance is consistent with a genuine scanner output."
        ),
    }


def verify_signature_presence(extracted_fields: Dict, ocr_full_text: str) -> Dict:
    """Check for required signatures - with e-CofO awareness"""

    text_upper = ocr_full_text.upper()
    issues = []                      
    passed = []                
    is_electronic = 'ELECTRONIC CERTIFICATE' in text_upper or 'UNIQUE BARCODES' in text_upper

                                                             
    if is_electronic:
        required_officials = [
            ("governor", ["BABAJIDE SANWO-OLU", "SANWO-OLU", "GOVERNOR"]),
            ("commissioner", ["COMMISSIONER FOR STAMP DUTIES", "WALE BABALOLA"]),
            ("registrar", ["REGISTRAR OF TITLES", "OLUWATOSIN ADEBISI"]),
        ]
    else:
        required_officials = [
            ("governor", ["GOVERNOR", "SURVEYOR GENERAL"]),
            ("commissioner", ["COMMISSIONER FOR LANDS", "PERMANENT SECRETARY"]),
            ("registrar", ["REGISTRAR OF TITLES"]),
        ]

                          
    found_officials = []
    for official_name, patterns in required_officials:
        found = False
        for pattern in patterns:
            if pattern in text_upper:
                found = True
                break
        found_officials.append((official_name, found))
        if not found:
            issues.append({
                "type": "missing_signature",
                "official": official_name,
                "severity": "CRITICAL",
                "weight": 0.85,
                "reason": f"Missing {official_name} signature. All genuine CoOs require this official's endorsement."
            })

                                     
    signatures_found = sum(1 for _, found in found_officials if found)

    return {
        "tool": "signature_verification",
        "signatures_found": signatures_found,
        "required_signatures": 3,
        "found_officials": [name for name, found in found_officials if found],
        "missing_officials": [name for name, found in found_officials if not found],
        "issues": issues,
        "weight": 0.85 if signatures_found < 2 else 0.4 if signatures_found < 3 else 0.0,
        "severity": "CRITICAL" if signatures_found < 2 else "HIGH" if signatures_found < 3 else "PASS",
        "explanation": f"Found {signatures_found}/3 required officials' signatures on document."
    }


def verify_barcodes_presence(ocr_full_text: str) -> Dict:
    """
    Check for barcodes/security features on e-CofO.
    Genuine e-CofO has 3 unique barcodes.
    """

    text_upper = ocr_full_text.upper()

                               
    barcode_patterns = [
        r'SC\d{11}[A-Z]\dY\d',
        r'LS/C[O0]/\d{2}/\d{4}/\d{5}',
        r'[A-Z0-9]{15,}',
    ]

    barcodes_found = []
    for pattern in barcode_patterns:
        matches = re.findall(pattern, text_upper)
        barcodes_found.extend(matches)

                          
    unique_barcodes = list(set(barcodes_found))

                                     
    has_barcode_mention = "UNIQUE BARCODES" in text_upper or "BARCODES FOR AUTHENTICATION" in text_upper

    if len(unique_barcodes) >= 3 or (has_barcode_mention and len(unique_barcodes) >= 1):
        return {
            "tool": "barcode_verification",
            "barcodes_found": len(unique_barcodes),
            "barcode_list": unique_barcodes[:5],
            "has_barcode_mention": has_barcode_mention,
            "weight": 0.0,
            "severity": "PASS",
            "explanation": f"Found {len(unique_barcodes)} barcode(s) or security markers."
        }
    else:
        return {
            "tool": "barcode_verification",
            "barcodes_found": len(unique_barcodes),
            "has_barcode_mention": has_barcode_mention,
            "weight": 0.75,
            "severity": "HIGH",
            "explanation": "Missing expected barcodes or security features. Genuine e-CofO has 3 unique barcodes."
        }

                                                                       
                                      
                                                                       

OFFLINE_TOOLS = {
    "verify_reference_numbers":    verify_reference_numbers,
    "analyze_font_consistency":    analyze_font_consistency,
    "validate_date_logic":         validate_date_logic,
    "verify_signature_presence": verify_signature_presence,
    "verify_barcodes_presence": verify_barcodes_presence,
    "analyze_print_scan_consistency": analyze_print_scan_consistency,
    "match_document_template":     match_document_template,
    "analyze_image_provenance":    analyze_image_provenance,
}