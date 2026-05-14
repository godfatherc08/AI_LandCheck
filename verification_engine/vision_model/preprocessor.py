                                            

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Prepares document images for better OCR and analysis"""

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skewed/rotated documents"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            coords = np.column_stack(np.where(gray > 0))

            if len(coords) < 10:
                return image

            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

            return image
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Improve contrast for faded documents"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove scanner noise or grain"""
        try:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except Exception as e:
            logger.warning(f"Noise removal failed: {e}")
            return image

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen blurry text"""
        try:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image

    def process(self, image: Image.Image) -> Image.Image:
        """Apply all preprocessing steps"""
        img_np = np.array(image)
        img_np = self.deskew(img_np)
        img_np = self.enhance_contrast(img_np)
        img_np = self.remove_noise(img_np)
        return Image.fromarray(img_np)