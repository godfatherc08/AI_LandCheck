"""
Updated OCR extraction using DocumentIntelligence pipeline
Zone-aware OCR + stamp/signature detection + cross-validation
"""

from verification_engine.vision_model.document_intelligence import DocumentIntelligence
from verification_engine.vision_model.forgery import ForgeryDetector
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import time


class LandVerifyOCR:
    """
    Enhanced OCR using DocumentIntelligence pipeline
    Combines zone-aware OCR with forgery detection
    """

    def __init__(self, languages: List[str] = None, enable_vision: bool = True,
                 sensitivity: str = 'medium'):
        self.enable_vision = enable_vision
        self.doc_intel = DocumentIntelligence()

        if enable_vision:
            self.forgery_detector = ForgeryDetector(sensitivity=sensitivity)
        else:
            self.forgery_detector = None

    def process_page(self, image: Image.Image, page_num: int = 1) -> Dict:
        """
        Process a single page using DocumentIntelligence pipeline

        Returns:
            Dict with:
            - extracted_fields: Structured fields from zone-aware extraction
            - zones: Zones detected
            - stamp_info: Stamp detection results
            - signature_info: Signature detection results
            - validation: Cross-field validation results
            - vision_forgery: Original forgery detection results
            - page_text: Combined OCR text
        """
        image_np = np.array(image)

                                           
        result = self.doc_intel.process(image_np)

                                             
        extracted_fields = {}
        for field_name, field_data in result.fields.items():
            extracted_fields[field_name] = {
                "value": field_data.cleaned_text,
                "confidence": field_data.confidence,
                "zone": field_data.zone,
                "page": page_num
            }

                                          
        vision_result = None
        if self.forgery_detector:
            vision_result = self.forgery_detector.analyze(image_np)

                                        
        page_text = " ".join([
            r[1] for zone_results in result.raw_ocr_by_zone.values()
            for r in zone_results
        ])

        return {
            'page_number': page_num,
            'page_text': page_text,
            'tokens': page_text.split(),
            'boxes': [],                                                       
            'image': image,
            'document_intelligence': {
                'fields': extracted_fields,
                'zones_found': result.zones_found,
                'stamp_detected': result.stamp_detected,
                'stamp_confidence': result.stamp_confidence,
                'signature_detected': result.signature_detected,
                'seal_detected': result.seal_detected,
                'document_orientation': result.document_orientation,
                'overall_ocr_confidence': result.overall_ocr_confidence,
                'warnings': result.warnings,
                'validation_passed': result.warnings == []
            },
            'vision_forgery': vision_result
        }

    def process_image_file(self, image_path: str) -> Dict:
        """Process a single image file"""
        from PIL import Image

        print(f"📸 Processing image: {image_path}")

                    
        image = Image.open(image_path)

                           
        result = self.process_page(image, page_num=1)

                                              
        if 'document_intelligence' in result:
            fields = result['document_intelligence'].get('fields', {})
            print(f"📋 Extracted {len(fields)} fields from document")
            for field_name, field_data in fields.items():
                print(f"   - {field_name}: {field_data.get('cleaned_text', field_data.get('value', 'N/A'))}")

        return result
    def process_pdf_pages(self, pdf_path: str, max_pages: Optional[int] = None) -> List[Dict]:
        """Process all pages of a PDF"""
        import fitz

        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages) if max_pages else len(doc)
        results = []

        for page_num in range(total_pages):
            page = doc[page_num]
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            page_result = self.process_page(image, page_num + 1)
            results.append(page_result)
            print(
                f"✓ Page {page_num + 1} processed (stamp: {page_result['document_intelligence']['stamp_detected']}, sig: {page_result['document_intelligence']['signature_detected']})")

        doc.close()
        return results

    def process_image_file(self, image_path: str) -> Dict:
        """Process a single image file"""
        image = Image.open(image_path)
        return self.process_page(image, 1)


                                               
LandVerifyOCRCompat = LandVerifyOCR