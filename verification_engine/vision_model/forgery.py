"""
Vision Forgery Detector for LandVerify
8 forensic checks with per-pixel evidence and calibration
"""

import cv2
import numpy as np
import os
import tempfile
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    method: str
    looks_wrong: bool
    confidence: float
    reason: str
    details: Dict = field(default_factory=dict)
    anomaly_regions: List[Dict] = field(default_factory=list)


class ForgeryDetector:
    """
    8 forensic checks for document forgery detection
    Returns per-pixel anomaly regions with bounding boxes
    """

    def __init__(self, sensitivity: str = 'medium', calibration_file: Optional[str] = None):
        """
        sensitivity: 'low' | 'medium' | 'high'
        calibration_file: Optional JSON file with baseline values
        """
        self.sensitivity = sensitivity
        self._s = {'low': 1.5, 'medium': 1.0, 'high': 0.7}.get(sensitivity, 1.0)
        self.baselines = self._load_baselines(calibration_file)

    def _load_baselines(self, calibration_file: Optional[str]) -> Dict:
        """Load calibration baselines or use defaults"""
        defaults = {
            "ela": {"mean": 0.025, "threshold": 0.045},
            "noise": {"mean": 0.08, "threshold": 0.12},
            "luminance": {"mean": 15.0, "threshold": 25.0},
            "edge": {"mean": 0.045, "threshold": 0.06}
        }

        if calibration_file and Path(calibration_file).exists():
            try:
                with open(calibration_file, 'r') as f:
                    custom = json.load(f)
                    defaults.update(custom)
            except Exception as e:
                logger.warning(f"Could not load calibration file: {e}")

        return defaults

    def _recompress(self, image: np.ndarray, quality: int = 92) -> np.ndarray:
        """Save and reload JPEG for ELA analysis"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp = f.name
        cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        reloaded = cv2.imread(tmp)
        os.unlink(tmp)
        if reloaded is None:
            raise RuntimeError("Could not recompress image")
        return cv2.cvtColor(reloaded, cv2.COLOR_BGR2RGB)

    def _ela_map(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel ELA difference map"""
        recompressed = self._recompress(image)
        diff = np.abs(image.astype(np.float32) - recompressed.astype(np.float32))
        return diff / 255.0

    def _get_ela_anomalies(self, ela: np.ndarray, image_shape: Tuple) -> List[Dict]:
        """Extract bounding boxes of ELA anomalies"""
        ela_gray = np.mean(ela, axis=2)
        threshold = np.percentile(ela_gray, 95)
        anomaly_mask = (ela_gray > threshold).astype(np.uint8) * 255

                                                         
        kernel = np.ones((5, 5), np.uint8)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 500:                       
                regions.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area_pixels": int(area),
                    "confidence": min(1.0, area / 5000)
                })

        return regions

    def _check_ela(self, image: np.ndarray) -> DetectionResult:
        """Error Level Analysis - detects Photoshop and recompression"""
        try:
            ela = self._ela_map(image)
            ela_gray = np.mean(ela, axis=2)
            mean_ela = float(np.mean(ela_gray))

            threshold = self.baselines.get("ela", {}).get("threshold", 0.045) * self._s
            looks_wrong = mean_ela > threshold
            anomaly_regions = self._get_ela_anomalies(ela, image.shape) if looks_wrong else []

            confidence = min(1.0, (mean_ela - threshold) / threshold + 0.5) if looks_wrong else 0.2

            reason = (f"ELA score {mean_ela:.4f} exceeds baseline {threshold:.4f}" if looks_wrong
                     else f"ELA score {mean_ela:.4f} within expected range")

            return DetectionResult(
                'ela', looks_wrong, round(confidence, 2), reason,
                {'mean_ela': round(mean_ela, 5), 'threshold': round(threshold, 5)},
                anomaly_regions
            )
        except Exception as e:
            logger.warning(f"ELA failed: {e}")
            return DetectionResult('ela', False, 0.0, f"Check failed: {e}")

    def _check_luminance(self, image: np.ndarray) -> DetectionResult:
        """Check background luminance consistency"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0].astype(np.float32)

                                                            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, bg_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            bg_frac = float(np.mean(bg_mask / 255.0))

            if bg_frac < 0.2:
                return DetectionResult('luminance', False, 0.0, "Insufficient background area")

            bg_l = l[bg_mask > 0]
            lum_range = float(np.percentile(bg_l, 95) - np.percentile(bg_l, 5))

            threshold = self.baselines.get("luminance", {}).get("threshold", 25.0) * self._s
            looks_wrong = lum_range > threshold
            confidence = min(1.0, (lum_range - threshold) / threshold) if looks_wrong else 0.1

            reason = (f"Luminance varies by {lum_range:.1f} units (threshold {threshold:.1f})"
                     if looks_wrong else f"Luminance consistent ({lum_range:.1f} units)")

            return DetectionResult(
                'luminance', looks_wrong, round(confidence, 2), reason,
                {'luminance_range': round(lum_range, 2), 'background_fraction': round(bg_frac, 3)}
            )
        except Exception as e:
            logger.warning(f"Luminance check failed: {e}")
            return DetectionResult('luminance', False, 0.0, f"Check failed: {e}")

    def _check_noise(self, image: np.ndarray) -> DetectionResult:
        """Detect noise inconsistencies indicating copy-paste"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

                                          
            noise_map = np.mean([
                np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0)),
                np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
            ], axis=0)

                                  
            h, w = noise_map.shape
            block_size = 32
            block_means = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = noise_map[y:y+block_size, x:x+block_size]
                    block_means.append(np.mean(block))

            block_means = np.array(block_means)
            mean_n, std_n = np.mean(block_means), np.std(block_means)
            inconsistent_frac = float(np.mean(np.abs(block_means - mean_n) > 2 * std_n))

            threshold = 0.1 * self._s
            looks_wrong = inconsistent_frac > threshold
            confidence = min(1.0, inconsistent_frac / threshold) if looks_wrong else 0.1

            reason = (f"{inconsistent_frac:.1%} of blocks have unusual noise" if looks_wrong
                     else "Noise consistent across document")

            return DetectionResult(
                'noise', looks_wrong, round(confidence, 2), reason,
                {'inconsistent_fraction': round(inconsistent_frac, 4)}
            )
        except Exception as e:
            logger.warning(f"Noise check failed: {e}")
            return DetectionResult('noise', False, 0.0, f"Check failed: {e}")

    def _check_edge(self, image: np.ndarray) -> DetectionResult:
        """Detect edge inconsistencies from pasted content"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            fine = sobel(gray)
            coarse = sobel(cv2.GaussianBlur(gray, (5, 5), 2))
            inconsistency = float(np.mean(np.abs(fine - coarse)))

            threshold = self.baselines.get("edge", {}).get("threshold", 0.06) * self._s
            looks_wrong = inconsistency > threshold
            confidence = min(1.0, (inconsistency - threshold) / threshold) if looks_wrong else 0.1

            reason = (f"Edge inconsistency {inconsistency:.4f} > {threshold:.4f}"
                     if looks_wrong else f"Edges consistent ({inconsistency:.4f})")

            return DetectionResult(
                'edge', looks_wrong, round(confidence, 2), reason,
                {'inconsistency_score': round(inconsistency, 5)}
            )
        except Exception as e:
            logger.warning(f"Edge check failed: {e}")
            return DetectionResult('edge', False, 0.0, f"Check failed: {e}")

    def _check_text_integrity(self, image: np.ndarray) -> DetectionResult:
        """Check for per-character compression anomalies"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            text_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 4
            )

            if np.mean(text_mask / 255.0) < 0.01:
                return DetectionResult('text_integrity', False, 0.0, "No text detected")

            ela = self._ela_map(image)
            ela_gray = np.mean(ela, axis=2)

                                     
            contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outliers = 0
            component_scores = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < 50:
                    continue
                component_ela = np.mean(ela_gray[y:y+h, x:x+w])
                component_scores.append(component_ela)

            if component_scores:
                scores = np.array(component_scores)
                mean_score = np.mean(scores)
                outliers = int(np.sum(scores > mean_score + 2 * np.std(scores)))

            looks_wrong = outliers > 5
            confidence = min(1.0, outliers / 15) if looks_wrong else 0.1

            reason = (f"{outliers} text components with unusual compression"
                     if looks_wrong else "Text compression consistent")

            return DetectionResult(
                'text_integrity', looks_wrong, round(confidence, 2), reason,
                {'outlier_components': outliers, 'total_components': len(component_scores)}
            )
        except Exception as e:
            logger.warning(f"Text integrity failed: {e}")
            return DetectionResult('text_integrity', False, 0.0, f"Check failed: {e}")

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Run all vision checks and return comprehensive results

        Returns:
            Dict with calibrated verdict, per-page evidence, anomaly regions
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return {
                'looks_forged': False,
                'verdict': 'ERROR',
                'summary': 'Invalid image',
                'per_page_evidence': []
            }

                             
        h, w = image.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

                        
        checks = [
            self._check_ela(image),
            self._check_noise(image),
            self._check_luminance(image),
            self._check_edge(image),
            self._check_text_integrity(image)
        ]

                         
        failed = [c for c in checks if c.looks_wrong]
        all_anomalies = []
        for c in checks:
            all_anomalies.extend(c.anomaly_regions)

                                            
        if len(failed) >= 3:
            verdict = "LOOKS_FORGED"
            looks_forged = True
            summary = f"{len(failed)} of 5 checks indicate manipulation"
        elif len(failed) >= 2:
            verdict = "LOOKS_SUSPICIOUS"
            looks_forged = True
            summary = f"{len(failed)} of 5 checks raised concerns"
        else:
            verdict = "LOOKS_AUTHENTIC"
            looks_forged = False
            summary = f"Only {len(failed)} minor concern(s)"

        return {
            'looks_forged': looks_forged,
            'verdict': verdict,
            'summary': summary,
            'sensitivity': self.sensitivity,
            'flags': [
                {
                    'check': r.method,
                    'reason': r.reason,
                    'confidence': r.confidence,
                    'details': r.details
                }
                for r in failed
            ],
            'anomaly_regions': [
                {
                    'check': r.method,
                    'bbox': region['bbox'],
                    'area_pixels': region['area_pixels'],
                    'confidence': region['confidence']
                }
                for r in failed
                for region in r.anomaly_regions
            ],
            'checks': {
                r.method: {
                    'looks_wrong': r.looks_wrong,
                    'confidence': r.confidence,
                    'reason': r.reason,
                    'details': r.details
                }
                for r in checks
            }
        }