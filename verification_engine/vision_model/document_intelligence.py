                                                     

import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import io


@dataclass
class Zone:
    name: str
    bbox: Tuple[int, int, int, int]                   
    image: np.ndarray
    zone_type: str                                                         


@dataclass
class DocumentField:
    field_name: str
    raw_text: str
    cleaned_text: str
    confidence: float
    zone: str
    bbox: List


@dataclass
class IntelligenceResult:
    fields: Dict[str, DocumentField]
    zones_found: List[str]
    stamp_detected: bool
    stamp_confidence: float
    signature_detected: bool
    seal_detected: bool
    document_orientation: float                                   
    overall_ocr_confidence: float
    warnings: List[str]
    raw_ocr_by_zone: Dict


class DocumentIntelligence:
    """
    Zone-aware document intelligence pipeline.
    Segments a certificate into structural zones before OCR,
    so field extraction is driven by spatial position not text order.
    """

                                              
    REF_PATTERNS = {
        'osg_ref':      r'OSG/\d+/\d+',
        'aros_plan':    r'AROS/LA/\d+/\d+/[A-Z]\d+/\d+',
        'ros_plan':     r'ROS/LA/\d+',
        'cof_number':   r'C\s*of\s*[Oo]\s*[-–]?\s*\d+',
        'luc_number':   r'LUC/\d+/\d+',
    }

                                     
    LAGOS_LGAS = {
        'agege', 'ajeromi-ifelodun', 'alimosho', 'amuwo-odofin',
        'apapa', 'badagry', 'epe', 'eti-osa', 'ibeju-lekki',
        'ifako-ijaiye', 'ikeja', 'ikorodu', 'kosofe', 'lagos island',
        'lagos mainland', 'mushin', 'ojo', 'oshodi-isolo', 'shomolu',
        'surulere',
    }

    def __init__(self):
        self._ocr = None

    def _get_ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

                                                                          
                              
                                                                          

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deskew, denoise, and normalise a scanned document.
        Returns corrected image + degrees corrected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                                                                
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

                                           
        skew_angle = self._detect_skew(denoised)
        if abs(skew_angle) > 0.3:
            image = self._rotate(image, skew_angle)

                                                              
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_corrected = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhanced = clahe.apply(gray_corrected)
        image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return image, skew_angle

    def _detect_skew(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not angles:
            return 0.0
                                       
        return float(np.median(angles))

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

                                                                          
                               
                                                                          

    def segment_zones(self, image: np.ndarray) -> List[Zone]:
        """
        Segment a certificate into structural zones.
        Uses a combination of whitespace analysis and
        visual feature detection (seal/stamp/signature regions).
        """
        h, w = image.shape[:2]
        zones = []

                                                                           
                                                                         
        zone_defs = [
            ("header",    (0,        0,        w,        int(h * 0.18))),
            ("ref_date",  (0,        int(h * 0.18), w,   int(h * 0.30))),
            ("body",      (0,        int(h * 0.30), w,   int(h * 0.72))),
            ("stamp_sig", (int(w * 0.45), int(h * 0.68), w, int(h * 0.88))),
            ("footer",    (0,        int(h * 0.88), w,   h)),
        ]

        for name, (x1, y1, x2, y2) in zone_defs:
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                zones.append(Zone(
                    name=name,
                    bbox=(x1, y1, x2, y2),
                    image=crop,
                    zone_type=name,
                ))

                                                                  
        stamp_zone = self._find_stamp_zone(image)
        if stamp_zone:
            zones.append(stamp_zone)

        return zones

    def _find_stamp_zone(self, image: np.ndarray) -> Optional[Zone]:
        """Find circular stamp regions using Hough circle detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=50,
            param1=100, param2=40,
            minRadius=30, maxRadius=int(min(h, w) * 0.15)
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
                                                            
        cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        pad = int(r * 0.3)
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(w, cx + r + pad)
        y2 = min(h, cy + r + pad)

        return Zone(
            name="detected_stamp",
            bbox=(x1, y1, x2, y2),
            image=image[y1:y2, x1:x2],
            zone_type="stamp",
        )

                                                                          
                            
                                                                          

    def ocr_zones(self, zones: List[Zone]) -> Dict[str, List]:
        """Run OCR per zone with zone-specific confidence thresholds."""
        ocr = self._get_ocr()
        results = {}

                                                      
        zone_configs = {
            "header":       {"scale": 2.0, "min_confidence": 0.6},
            "ref_date":     {"scale": 2.5, "min_confidence": 0.7},
            "body":         {"scale": 2.0, "min_confidence": 0.6},
            "stamp_sig":    {"scale": 3.0, "min_confidence": 0.4},                   
            "detected_stamp": {"scale": 3.0, "min_confidence": 0.3},
            "footer":       {"scale": 2.0, "min_confidence": 0.6},
        }

        for zone in zones:
            cfg = zone_configs.get(zone.name,
                                   {"scale": 2.0, "min_confidence": 0.6})

                                                  
            scaled = cv2.resize(
                zone.image,
                None,
                fx=cfg["scale"],
                fy=cfg["scale"],
                interpolation=cv2.INTER_CUBIC
            )

            result, _ = ocr(scaled)
            if result:
                filtered = [
                    r for r in result
                    if float(r[2]) >= cfg["min_confidence"]
                ]
                results[zone.name] = filtered
            else:
                results[zone.name] = []

        return results

    def verify_signatures(self, image: np.ndarray) -> Dict:
        """Verify signatures (flexible count for different C of O formats)"""
        signature_zone = image[int(image.shape[0] * 0.85):, :]              

        gray = cv2.cvtColor(signature_zone, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                             
        signatures = [c for c in contours if 500 < cv2.contourArea(c) < 5000]

                                                             
        return {
            "signatures_detected": len(signatures),
            "expected_signatures": 3,
            "all_present": len(signatures) >= 2,                
            "signature_positions": [cv2.boundingRect(c) for c in signatures[:3]]
        }
                                                                          
                                    
                                                                          

    def detect_stamp(self, image: np.ndarray) -> Dict:
        """
        Detect government stamps and seals.
        Checks for:
        - Circular/oval shapes (government seals)
        - Blue/red ink regions (common stamp colours)
        - Text within circular regions
        """
        h, w = image.shape[:2]

                                            
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                                                                
        blue_mask = cv2.inRange(hsv,
                                np.array([100, 50, 50]),
                                np.array([130, 255, 255]))
                       
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]),
                                np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]),
                                np.array([180, 255, 255]))
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

        blue_fraction = float(np.mean(blue_mask > 0))
        red_fraction  = float(np.mean(red_mask > 0))
        has_stamp_ink = blue_fraction > 0.01 or red_fraction > 0.01

                                     
        gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=50,
            param1=100, param2=40,
            minRadius=30, maxRadius=int(min(h, w) * 0.15)
        )
        has_circle = circles is not None

                                  
        confidence = 0.0
        if has_circle:   confidence += 0.5
        if has_stamp_ink: confidence += 0.4
        if has_circle and has_stamp_ink: confidence += 0.1                  

        return {
            "stamp_detected":  confidence > 0.4,
            "confidence":       round(confidence, 2),
            "has_circular_shape": has_circle,
            "has_stamp_ink":    has_stamp_ink,
            "blue_ink_fraction": round(blue_fraction, 4),
            "red_ink_fraction":  round(red_fraction, 4),
            "circle_count":     len(circles[0]) if circles is not None else 0,
        }

                                                                          
                                 
                                                                          

    def detect_signature(self, image: np.ndarray) -> Dict:
        """
        Detect handwritten signatures.
        Signatures have high local variance, irregular strokes,
        and appear in the lower-right quadrant of certificates.
        """
        h, w = image.shape[:2]

                                                             
        roi = image[int(h * 0.6):, int(w * 0.4):]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                                                                 
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"signature_detected": False, "confidence": 0.0}

                                                               
        sig_contours = [
            c for c in contours
            if 50 < cv2.contourArea(c) < (roi.shape[0] * roi.shape[1] * 0.3)
        ]

                                                                         
        irregularity_scores = []
        for c in sig_contours[:10]:
            area     = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                                                                           
                circularity = 4 * np.pi * area / (perimeter ** 2)
                irregularity_scores.append(1.0 - circularity)

        mean_irregularity = np.mean(irregularity_scores) if irregularity_scores else 0
        contour_density   = len(sig_contours) / max(len(contours), 1)

                                                            
        confidence = float(np.clip(
            0.5 * min(len(sig_contours) / 5, 1.0) +
            0.5 * mean_irregularity,
            0, 1
        ))

        return {
            "signature_detected": confidence > 0.35,
            "confidence": round(confidence, 2),
            "contour_count": len(sig_contours),
            "mean_irregularity": round(float(mean_irregularity), 3),
        }

                                                                          
                                         
                                                                      
                                                                          

    def extract_fields(self, ocr_by_zone: Dict, zones: List[Zone]) -> Dict[str, DocumentField]:
        """
        Map OCR text to document fields using spatial position within zones.
        """
        fields = {}
        zone_map = {z.name: z for z in zones}

                                                            
        body_results = ocr_by_zone.get("body", [])
        fields.update(self._extract_body_fields(body_results))

                                    
        ref_zone_text = self._zone_text(ocr_by_zone.get("ref_date", []))
        fields.update(self._extract_ref_fields(ref_zone_text, ocr_by_zone.get("ref_date", [])))

                                  
        header_text = self._zone_text(ocr_by_zone.get("header", []))
        if header_text and "issuing_authority" not in fields:
            fields["issuing_authority"] = DocumentField(
                field_name="issuing_authority",
                raw_text=header_text,
                cleaned_text=self._clean_text(header_text),
                confidence=self._zone_confidence(ocr_by_zone.get("header", [])),
                zone="header",
                bbox=[],
            )

        return fields
    def _extract_ref_fields(self, text: str, results: List) -> Dict:
        fields = {}

                                           
        for pattern_name, pattern in self.REF_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["ref_number"] = DocumentField(
                    field_name="ref_number",
                    raw_text=match.group(),
                    cleaned_text=match.group().upper().replace(" ", ""),
                    confidence=0.95,
                    zone="ref_date",
                    bbox=[],
                )
                break

                                                        
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{1,2}\s+\w+\s+\d{4}',
        ]
        for dp in date_patterns:
            match = re.search(dp, text)
            if match:
                fields["issue_date"] = DocumentField(
                    field_name="issue_date",
                    raw_text=match.group(),
                    cleaned_text=match.group(),
                    confidence=0.90,
                    zone="ref_date",
                    bbox=[],
                )
                break

        return fields

    def _extract_body_fields(self, results: List) -> Dict:
        """Extract fields from REAL Certificate of Occupancy"""
        fields = {}
        if not results:
            return fields

        full_text = " ".join(r[1] for r in results)

                                                      
                                               
                                                      

                                                       
        cof_patterns = [
            r'CO\s*NO\.?\s*:?\s*([A-Z0-9/]+)',                                
            r'C\s*OF\s*O\s*NO\.?\s*:?\s*([A-Z0-9/]+)',                                    
            r'LS/CO/\d{2}/\d{4}/\d{5}',                                    
            r'([A-Z]{2}/[A-Z]{2}/\d{2}/\d{4}/\d{5})',                              
            r'CO\s*N[Oo][.:]?\s*([A-Z0-9/]+)',                 
        ]
        for pattern in cof_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                fields["cof_number"] = DocumentField(
                    field_name="cof_number",
                    raw_text=value,
                    cleaned_text=value,
                    confidence=0.95,
                    zone="body",
                    bbox=[]
                )
                break

                                                                             
        term_patterns = [
            r'(\d{2,3})\s*YEARS?',
            r'NINETY[- ]NINE\s*\(?99\)?\s*YEARS?',
            r'TERM\s+(\d{2,3})',
        ]
        for pattern in term_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                term_value = "99" if 'NINETY' in full_text.upper() else match.group(1)
                fields["term_years"] = DocumentField(
                    field_name="term_years",
                    raw_text=term_value,
                    cleaned_text=term_value,
                    confidence=0.90,
                    zone="body",
                    bbox=[]
                )
                break

                                             
        purpose_match = re.search(r'FOR\s+([A-Z][A-Z\s]+?)\s+PURPOSE', full_text, re.IGNORECASE)
        if purpose_match:
            fields["land_use"] = DocumentField(
                field_name="land_use",
                raw_text=purpose_match.group(1),
                cleaned_text=purpose_match.group(1).strip().title(),
                confidence=0.85,
                zone="body",
                bbox=[]
            )

                                                                     
        owner_match = re.search(r'grants to the holder\(s\)\s+([A-Z][A-Z\s]+?)(?:\s+CONo|$)', full_text, re.IGNORECASE)
        if not owner_match:
            owner_match = re.search(r'([A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,})', full_text)
        if owner_match:
            fields["property_owner"] = DocumentField(
                field_name="property_owner",
                raw_text=owner_match.group(1),
                cleaned_text=owner_match.group(1).strip(),
                confidence=0.80,
                zone="body",
                bbox=[]
            )

                                      
        date_match = re.search(r'Date of Grant\s*(\d{2}/\d{2}/\d{4})', full_text, re.IGNORECASE)
        if not date_match:
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', full_text)
        if date_match:
            fields["issue_date"] = DocumentField(
                field_name="issue_date",
                raw_text=date_match.group(1),
                cleaned_text=date_match.group(1),
                confidence=0.90,
                zone="body",
                bbox=[]
            )

                        
        plot_match = re.search(r'PLOT\s*NO[.:]?\s*(\d+)', full_text, re.IGNORECASE)
        if plot_match:
            fields["plot_number"] = DocumentField(
                field_name="plot_number",
                raw_text=plot_match.group(1),
                cleaned_text=plot_match.group(1),
                confidence=0.85,
                zone="body",
                bbox=[]
            )

                
        lga_match = re.search(r'LOCAL\s+GOVERNMENT\s*:?\s*([A-Z][A-Z-]+)', full_text, re.IGNORECASE)
        if lga_match:
            fields["lga"] = DocumentField(
                field_name="lga",
                raw_text=lga_match.group(1),
                cleaned_text=lga_match.group(1).strip(),
                confidence=0.90,
                zone="body",
                bbox=[]
            )

                 
        area_match = re.search(r'AREA\s*:?\s*([\d,]+\.?\d*)\s*SQUARE\s+METRES?', full_text, re.IGNORECASE)
        if area_match:
            fields["plot_area_sqm"] = DocumentField(
                field_name="plot_area_sqm",
                raw_text=area_match.group(1),
                cleaned_text=area_match.group(1).replace(',', ''),
                confidence=0.85,
                zone="body",
                bbox=[]
            )

        print(f"📋 Extracted {len(fields)} fields from C of O")
        return fields
                                                                          
                                    
                                                                          

    def validate_fields(self, fields: Dict[str, DocumentField],
                        stamp_result: Dict,
                        signature_result: Dict) -> Dict:
        """
        Cross-validate extracted fields against known patterns
        for Lagos State Certificates of Occupancy.
        Flags inconsistencies that a forger would commonly introduce.
        """
        warnings = []
        passed   = []

                                 
        ref = fields.get("ref_number")
        if ref:
            if re.match(r'OSG/\d{3}/\d{5}', ref.cleaned_text):
                passed.append("OSG ref number format is valid")
            else:
                warnings.append(
                    f"Ref number '{ref.cleaned_text}' does not match "
                    f"expected OSG/XXX/XXXXX format"
                )
        else:
            warnings.append("No reference number detected")

                           
        date = fields.get("issue_date")
        if date:
            try:
                for fmt in ["%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(date.cleaned_text, fmt)
                        if dt.year < 1978:
                            warnings.append(
                                f"Issue date {date.cleaned_text} predates "
                                f"the Land Use Act 1978 — impossible for a CoO"
                            )
                        elif dt.year > datetime.now().year:
                            warnings.append("Issue date is in the future")
                        else:
                            passed.append(f"Issue date {date.cleaned_text} is plausible")
                        break
                    except ValueError:
                        continue
            except Exception:
                warnings.append("Could not parse issue date")

                              
        if not stamp_result.get("stamp_detected"):
            warnings.append(
                "No government stamp detected — all Lagos State CoOs carry "
                "an official Surveyor General stamp"
            )
        else:
            passed.append(f"Government stamp detected "
                          f"(confidence {stamp_result['confidence']:.0%})")

                                  
        if not signature_result.get("signature_detected"):
            warnings.append("No signature detected in signing zone")
        else:
            passed.append("Signature detected in expected position")

                               
        lga   = fields.get("lga")
        addr  = fields.get("address")
                             
        if lga and addr:
                                                                           
            if len(addr.cleaned_text) > 20 and lga.cleaned_text.lower() not in addr.cleaned_text.lower():
                                                                          
                garbled_indicators = ['DETAILS', 'USER', 'DATE', 'TRANSACTION', 'BLOCK']
                if not any(ind in addr.cleaned_text.upper() for ind in garbled_indicators):
                    warnings.append(f"LGA '{lga.cleaned_text}' not mentioned in address — possible field mismatch")

        return {
            "passed":   passed,
            "warnings": warnings,
            "field_count": len(fields),
            "clean_fields": {
                k: {"value": v.cleaned_text, "confidence": v.confidence, "zone": v.zone}
                for k, v in fields.items()
            },
        }

                                                                          
             
                                                                          

    def _zone_text(self, results: List) -> str:
        return " ".join(r[1] for r in results) if results else ""

    def _zone_confidence(self, results: List) -> float:
        if not results:
            return 0.0
        return float(np.mean([float(r[2]) for r in results]))

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def _clean_name(self, text: str) -> str:
                                                
        cleaned = re.sub(r'[^A-Za-z\s&\-]', '', text)
        return re.sub(r'\s+', ' ', cleaned).strip().upper()

                                     
    def detect_barcodes(self, image: np.ndarray) -> Dict:
        """
        Detect 2D barcodes (Data Matrix, QR codes) on the document.
        Real Lagos C of O has encrypted 2D barcodes for security.
        """
        from pyzbar import pyzbar

        barcodes = pyzbar.decode(image)
        results = []

        for barcode in barcodes:
            results.append({
                "type": barcode.type,
                "data": barcode.data.decode('utf-8') if barcode.data else None,
                "location": barcode.rect
            })

        return {
            "barcodes_detected": len(results) > 0,
            "barcode_count": len(results),
            "barcodes": results,
            "expected_count": 4,                                                         
            "complete_set": len(results) == 4
        }

                                                                          
                   
                                                                          

    def process(self, image: np.ndarray) -> IntelligenceResult:
        """
        Full pipeline: image in → structured result out.
        """
                       
        image, skew_angle = self.preprocess(image)

                          
        zones = self.segment_zones(image)

                         
        ocr_by_zone = self.ocr_zones(zones)

                            
        stamp_result = self.detect_stamp(image)

                                
        sig_result = self.detect_signature(image)

                           
        fields = self.extract_fields(ocr_by_zone, zones)

                     
        validation = self.validate_fields(fields, stamp_result, sig_result)

                                
        all_confidences = [
            float(r[2])
            for zone_results in ocr_by_zone.values()
            for r in zone_results
        ]
        overall_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

        return IntelligenceResult(
            fields=fields,
            zones_found=[z.name for z in zones],
            stamp_detected=stamp_result["stamp_detected"],
            stamp_confidence=stamp_result["confidence"],
            signature_detected=sig_result["signature_detected"],
            seal_detected=stamp_result["has_circular_shape"],
            document_orientation=skew_angle,
            overall_ocr_confidence=round(overall_confidence, 3),
            warnings=validation["warnings"],
            raw_ocr_by_zone=ocr_by_zone,
        )