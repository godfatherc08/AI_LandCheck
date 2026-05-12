from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import io


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[List[int]]
    page_num: int


class LandVerifyOCR:

    def __init__(self, languages: List[str] = None):
        if languages is None:
            languages = ['en']
        self.languages = languages
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            # OCR: use_angle_cls catches rotated text, show_log silences spam
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

    def process_image_array(self, image: Image.Image, page_num: int = 1) -> List[OCRResult]:
        reader = self._get_reader()
        image_np = np.array(image)
        result, _ = reader(image_np)  # ← call reader() directly, not reader.ocr()
        return self.convert_to_ocr_results(result or [], page_num)

    def process_image_file(self, image_path: str) -> List[OCRResult]:
        reader = self._get_reader()
        result, _ = reader(image_path)  # ← same here
        return self.convert_to_ocr_results(result or [], page_num=1)

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
        reader = self._get_reader()  # get the instance once
        images = images_data['images']
        page_results = {}

        def ocr_page(page):
            image_np = np.array(page['image'])
            result, _ = reader(image_np)  # pass reader directly, no self._get_reader() call
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
    def get_layoutlmv3_format(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict:
        images_data = self.extract_images_from_pdf(pdf_path, max_pages)

        # Use parallel processing instead of sequential
        parallel_result = self.process_pages_parallel(images_data)

        tokens, boxes, confidences, page_texts = [], [], [], {}

        for res in parallel_result['ocr_results']:
            tokens.append(res.text)
            confidences.append(res.confidence)
            x1, y1 = res.bbox[0]
            x2, y2 = res.bbox[2]
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            page_texts.setdefault(res.page_num, []).append(res.text)

        full_text = '\n\n'.join(
            ' '.join(page_texts[p]) for p in sorted(page_texts)
        )

        return {'tokens': tokens, 'boxes': boxes, 'confidences': confidences, 'full_text': full_text}

    def process_pages_streaming(self, pdf_path: str, max_pages: Optional[int] = None,
                                max_workers: int = 4) -> Generator[Dict, None, None]:
        """
        Process PDF pages in parallel and YIELD results as each page completes.
        This allows LayoutLMv3 to start processing immediately.

        Yields:
            Dict with page_number, tokens, boxes, confidences, page_text
        """
        # Extract images first
        images_data = self.extract_images_from_pdf(pdf_path, max_pages)
        images = images_data['images']
        reader = self._get_reader()

        def ocr_page(page):
            """OCR a single page and return results"""
            image_np = np.array(page['image'])
            result, _ = reader(image_np)
            ocr_results = self.convert_to_ocr_results(result or [], page['page_number'])

            # Convert to LayoutLMv3 format immediately
            tokens = []
            boxes = []
            confidences = []

            for res in ocr_results:
                tokens.append(res.text)
                confidences.append(res.confidence)
                x1, y1 = res.bbox[0]
                x2, y2 = res.bbox[2]
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

            return {
                'page_number': page['page_number'],
                'tokens': tokens,
                'boxes': boxes,
                'confidences': confidences,
                'page_text': ' '.join(tokens),
                'status': 'success'
            }

        # Submit all pages and yield as they complete
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(ocr_page, page): page
                for page in images
            }

            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    result = future.result()
                    print(f"✓ Page {page['page_number']} ready for LayoutLMv3")
                    yield result  # ← YIELD immediately, don't wait for other pages
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

    def get_layoutlmv3_format_streaming(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict:
        """
        Process PDF and return combined results (waits for all pages).
        Use this when you want the final complete result.
        """
        all_tokens = []
        all_boxes = []
        all_confidences = []
        all_page_texts = {}

        # Process pages as they complete
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
