# verification_engine/OCR/extraction.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import io
import sys
from pathlib import Path

# Add vision module path
sys.path.append(str(Path(__file__).parent.parent))
from verification_engine.vision_model.forgery import  ForgeryDetector
from verification_engine.vision_model.preprocessor import DocumentPreprocessor


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[List[int]]
    page_num: int


class LandVerifyOCR:

    def __init__(self, languages: List[str] = None, enable_vision: bool = True,
                 sensitivity: str = 'high'):
        """
        Initialize OCR engine with optional vision preprocessing

        Args:
            languages: List of language codes
            enable_vision: Whether to run vision checks on pages
            sensitivity: 'low', 'medium', or 'high' for forgery detection
        """
        if languages is None:
            languages = ['en']
        self.languages = languages
        self._reader = None
        self.enable_vision = enable_vision

        # Initialize vision components if enabled
        if enable_vision:
            self.forgery_detector = ForgeryDetector(sensitivity=sensitivity)
            self.preprocessor = DocumentPreprocessor()
        else:
            self.forgery_detector = None
            self.preprocessor = None

    def _get_reader(self):
        if self._reader is None:
            self._reader = RapidOCR()
        return self._reader

    def extract_images_from_pdf(self, pdf_path: str, max_pages: Optional[int] = None):
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages) if max_pages else len(doc)
        page_images = []

        for page_num in range(total_pages):
            page = doc[page_num]
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_images.append({
                'page_number': page_num + 1,
                'image': image,
                'size': image.size,
                'mode': image.mode
            })

        doc.close()
        return {'images': page_images, 'total_pages': total_pages, 'source_pdf': pdf_path}

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply vision preprocessing to enhance image quality"""
        if self.preprocessor is None:
            return image

        try:
            img_np = np.array(image)
            img_np = self.preprocessor.deskew(img_np)
            img_np = self.preprocessor.enhance_contrast(img_np)
            img_np = self.preprocessor.remove_noise(img_np)
            return Image.fromarray(img_np)
        except Exception as e:
            print(f"Warning: Preprocessing failed: {e}")
            return image

    def process_image_array(self, image: Image.Image, page_num: int = 1,
                            run_vision: bool = True) -> tuple:
        """
        Process a PIL Image object with optional vision checks

        Returns:
            (ocr_results, vision_result) tuple
        """
        reader = self._get_reader()

        # Apply preprocessing
        enhanced_image = self.preprocess_image(image)

        # Run OCR on enhanced image
        image_np = np.array(enhanced_image)
        result, _ = reader(image_np)
        ocr_results = self.convert_to_ocr_results(result or [], page_num)

        # Run vision forgery detection if enabled
        vision_result = None
        if run_vision and self.enable_vision:
            vision_result = self.forgery_detector.analyze(image_np)

        return ocr_results, vision_result

    def process_image_file(self, image_path: str) -> tuple:
        """Process a single image file with vision checks"""
        reader = self._get_reader()

        # Load image
        image = Image.open(image_path)
        enhanced_image = self.preprocess_image(image)

        # OCR
        image_np = np.array(enhanced_image)
        result, _ = reader(image_np)
        ocr_results = self.convert_to_ocr_results(result or [], page_num=1)

        # Vision
        vision_result = None
        if self.enable_vision:
            vision_result = self.forgery_detector.analyze(image_np)

        return ocr_results, vision_result

    def convert_to_ocr_results(self, rapid_output, page_num: int) -> List[OCRResult]:
        results = []
        if not rapid_output:
            return results
        for line in rapid_output:
            bbox, text, confidence = line[0], line[1], line[2]
            results.append(OCRResult(
                text=text,
                confidence=float(confidence),
                bbox=bbox,
                page_num=page_num
            ))
        return results

    def process_pages_parallel(self, images_data: Dict, max_workers: int = 4) -> Dict:
        reader = self._get_reader()
        images = images_data['images']
        page_results = {}

        def ocr_page(page):
            # Apply preprocessing
            enhanced = self.preprocess_image(page['image'])
            image_np = np.array(enhanced)
            result, _ = reader(image_np)
            return self.convert_to_ocr_results(result or [], page['page_number'])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(ocr_page, page): page
                for page in images
            }
            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    page_results[page['page_number']] = future.result()
                    print(f"✓ Page {page['page_number']}")
                except Exception as e:
                    print(f"✗ Page {page['page_number']}: {e}")
                    page_results[page['page_number']] = []

        all_ocr_results = []
        for page_num in sorted(page_results):
            all_ocr_results.extend(page_results[page_num])

        return {'ocr_results': all_ocr_results}

    def process_pages_streaming(self, pdf_path: str, max_pages: Optional[int] = None,
                                max_workers: int = 4) -> Generator[Dict, None, None]:
        """
        Process PDF pages in parallel and YIELD results as each page completes.
        Includes vision analysis if enabled.
        """
        images_data = self.extract_images_from_pdf(pdf_path, max_pages)
        images = images_data['images']
        reader = self._get_reader()

        def process_page(page):
            # Apply preprocessing
            enhanced = self.preprocess_image(page['image'])
            image_np = np.array(enhanced)

            # OCR
            result, _ = reader(image_np)
            ocr_results = self.convert_to_ocr_results(result or [], page['page_number'])

            # Build tokens and boxes
            tokens = []
            boxes = []
            confidences = []

            for res in ocr_results:
                tokens.append(res.text)
                confidences.append(res.confidence)
                x1, y1 = res.bbox[0]
                x2, y2 = res.bbox[2]
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

            page_data = {
                'page_number': page['page_number'],
                'tokens': tokens,
                'boxes': boxes,
                'confidences': confidences,
                'page_text': ' '.join(tokens),
                'status': 'success'
            }

            # Add vision results if enabled
            if self.enable_vision:
                vision_result = self.forgery_detector.analyze(image_np)
                page_data['vision'] = vision_result

            return page_data

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(process_page, page): page
                for page in images
            }

            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    result = future.result()
                    if self.enable_vision and result.get('vision', {}).get('looks_forged'):
                        print(f"⚠️ VISION ALERT: Page {page['page_number']} shows forgery indicators")
                    print(f"✓ Page {page['page_number']} ready")
                    yield result
                except Exception as e:
                    print(f"✗ Page {page['page_number']} failed: {e}")
                    yield {
                        'page_number': page['page_number'],
                        'tokens': [],
                        'boxes': [],
                        'confidences': [],
                        'page_text': '',
                        'status': 'failed',
                        'error': str(e)
                    }

    def get_layoutlmv3_format(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict:
        all_tokens = []
        all_boxes = []
        all_confidences = []
        all_page_texts = {}

        for page_result in self.process_pages_streaming(pdf_path, max_pages):
            page_num = page_result['page_number']
            all_tokens.extend(page_result['tokens'])
            all_boxes.extend(page_result['boxes'])
            all_confidences.extend(page_result['confidences'])
            all_page_texts[page_num] = page_result['page_text']

        full_text = '\n\n'.join(all_page_texts[p] for p in sorted(all_page_texts))

        return {
            'tokens': all_tokens,
            'boxes': all_boxes,
            'confidences': all_confidences,
            'full_text': full_text
        }

    def get_full_analysis(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict:
        """
        Complete analysis: OCR + Vision + LayoutLMv3-ready format
        """
        all_page_data = []
        all_tokens = []
        all_boxes = []
        all_confidences = []
        all_page_texts = {}
        vision_summary = {
            'pages_analyzed': 0,
            'forged_pages': [],
            'suspicious_pages': [],
            'all_flags': []
        }

        for page_result in self.process_pages_streaming(pdf_path, max_pages):
            page_num = page_result['page_number']
            all_page_data.append(page_result)
            all_tokens.extend(page_result['tokens'])
            all_boxes.extend(page_result['boxes'])
            all_confidences.extend(page_result['confidences'])
            all_page_texts[page_num] = page_result['page_text']

            # Collect vision results
            if 'vision' in page_result:
                vision_summary['pages_analyzed'] += 1
                if page_result['vision']['looks_forged']:
                    vision_summary['forged_pages'].append(page_num)
                    vision_summary['all_flags'].extend(page_result['vision']['flags'])
                elif page_result['vision']['verdict'] == 'LOOKS SUSPICIOUS':
                    vision_summary['suspicious_pages'].append(page_num)

        full_text = '\n\n'.join(all_page_texts[p] for p in sorted(all_page_texts))

        return {
            'tokens': all_tokens,
            'boxes': all_boxes,
            'confidences': all_confidences,
            'full_text': full_text,
            'page_data': all_page_data,
            'vision_summary': vision_summary if self.enable_vision else None
        }