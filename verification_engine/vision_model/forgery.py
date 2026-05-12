# verification_engine/vision/forgery_detector.py

import cv2
import numpy as np
import os
import tempfile
from typing import Dict, List
from dataclasses import dataclass, field
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    method: str
    looks_wrong: bool
    confidence: float      # 0.0 = not sure, 1.0 = very sure
    reason: str            # human-readable explanation
    details: Dict = field(default_factory=dict)


class ForgeryDetector:
    """
    Answers one question: does this Certificate of Occupancy LOOK visually wrong?

    Eight visual checks, each looking for something a human expert would
    notice on close inspection — inconsistent noise, suspicious compression
    artifacts, duplicated regions, unnatural edges, and so on.
    """

    def __init__(self, sensitivity: str = 'high'):
        """
        sensitivity: 'low' | 'medium' | 'high'
        Use 'high' for legal documents — errs toward flagging over missing things.
        """
        self.sensitivity = sensitivity
        # Multiplier makes thresholds tighter at high sensitivity
        self._s = {'low': 1.5, 'medium': 1.0, 'high': 0.7}.get(sensitivity, 1.0)

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    def _recompress(self, image: np.ndarray, quality: int = 92) -> np.ndarray:
        """Save image as JPEG and reload — used by ELA and text integrity."""
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
        """Per-pixel ELA difference map, normalised 0-1."""
        recompressed = self._recompress(image)
        diff = np.abs(image.astype(np.float32) - recompressed.astype(np.float32))
        return diff / 255.0

    # ------------------------------------------------------------------ #
    # 1. ERROR LEVEL ANALYSIS
    # Does the document have localised JPEG compression hotspots?
    # Genuine docs recompress uniformly. Edited regions stand out.
    # ------------------------------------------------------------------ #

    def _check_ela(self, image: np.ndarray) -> DetectionResult:
        try:
            ela = self._ela_map(image)
            mean_ela    = float(np.mean(ela))
            high_ela    = float(np.percentile(ela, 95))
            hotspot_frac = float(np.mean(ela[:, :, 0] > mean_ela * 3 + 0.02))

            threshold_mean    = 0.03  * self._s
            threshold_high    = 0.15  * self._s
            threshold_hotspot = 0.03  * self._s

            looks_wrong = (
                mean_ela    > threshold_mean or
                high_ela    > threshold_high or
                hotspot_frac > threshold_hotspot
            )

            if looks_wrong:
                if hotspot_frac > threshold_hotspot:
                    reason = (f"Localised compression hotspots cover {hotspot_frac:.1%} of the document — "
                              f"typical of edited text fields or pasted stamps.")
                elif high_ela > threshold_high:
                    reason = (f"Top-5% ELA value is {high_ela:.4f} (expected < {threshold_high:.4f}) — "
                              f"specific regions were re-saved at a different quality.")
                else:
                    reason = (f"Overall ELA level {mean_ela:.4f} exceeds expected baseline — "
                              f"document may have been opened and re-saved after editing.")
            else:
                reason = "Compression residuals are uniform — no editing artifacts detected."

            confidence = min(1.0, max(mean_ela / threshold_mean,
                                      high_ela / threshold_high,
                                      hotspot_frac / max(threshold_hotspot, 1e-8)) / 2)

            return DetectionResult('ela', looks_wrong, round(confidence, 2), reason,
                                   {'mean_ela': round(mean_ela, 5),
                                    'high_percentile_ela': round(high_ela, 5),
                                    'hotspot_fraction': round(hotspot_frac, 4)})
        except Exception as e:
            logger.warning(f"ELA failed: {e}")
            return DetectionResult('ela', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 2. NOISE INCONSISTENCY
    # Is the background noise uniform across the whole page?
    # A spliced-in region brings its own noise floor from its source scan.
    # ------------------------------------------------------------------ #

    def _check_noise(self, image: np.ndarray) -> DetectionResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Multi-scale noise: average residual at three blur radii
            noise_map = np.mean([
                np.abs(gray - cv2.GaussianBlur(gray, (k, k), 0))
                for k in [3, 5, 7]
            ], axis=0)

            # Measure noise per 32×32 block
            block_size = 32
            h, w = noise_map.shape
            block_means = [
                np.mean(noise_map[y:y+block_size, x:x+block_size])
                for y in range(0, h - block_size, block_size)
                for x in range(0, w - block_size, block_size)
            ]
            block_means = np.array(block_means)
            mean_n, std_n = np.mean(block_means), np.std(block_means)

            z_scores = np.abs((block_means - mean_n) / (std_n + 1e-8))
            inconsistent_frac = float(np.mean(z_scores > 2.5))
            threshold = 0.04 * self._s

            looks_wrong = inconsistent_frac > threshold
            reason = (
                f"{inconsistent_frac:.1%} of page blocks have noise levels that don't match "
                f"the rest of the document — suggests content from a different source was inserted."
                if looks_wrong else
                "Noise is consistent across the page — no splicing detected."
            )

            return DetectionResult('noise', looks_wrong, round(min(inconsistent_frac / threshold, 1.0), 2),
                                   reason,
                                   {'inconsistent_block_fraction': round(inconsistent_frac, 4),
                                    'noise_mean': round(float(mean_n), 4),
                                    'noise_std': round(float(std_n), 4)})
        except Exception as e:
            logger.warning(f"Noise check failed: {e}")
            return DetectionResult('noise', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 3. DCT BLOCK ANALYSIS
    # Are JPEG compression blocks internally consistent?
    # A region saved at a different quality or from a different image
    # shows up as DCT outlier blocks.
    # ------------------------------------------------------------------ #

    def _check_dct(self, image: np.ndarray) -> DetectionResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape

            hf_ratios = []
            for y in range(0, h - 8, 8):
                for x in range(0, w - 8, 8):
                    block = gray[y:y+8, x:x+8]
                    dct   = cv2.dct(block)
                    total = np.sum(dct ** 2) + 1e-8
                    hf    = np.sum(dct[4:, 4:] ** 2)
                    hf_ratios.append(hf / total)

            hf_ratios  = np.array(hf_ratios)
            hf_mean    = float(np.mean(hf_ratios))
            hf_std     = float(np.std(hf_ratios))
            outlier_frac = float(np.mean(hf_ratios > hf_mean + 2.5 * hf_std))

            threshold = 0.05 * self._s
            looks_wrong = outlier_frac > threshold
            reason = (
                f"{outlier_frac:.1%} of 8×8 DCT blocks have anomalous high-frequency energy — "
                f"these blocks were likely compressed independently (pasted from another document)."
                if looks_wrong else
                "DCT block energy is consistent — no compression-origin mismatch found."
            )

            return DetectionResult('dct', looks_wrong, round(min(outlier_frac / threshold, 1.0), 2),
                                   reason,
                                   {'outlier_block_fraction': round(outlier_frac, 4),
                                    'hf_mean': round(hf_mean, 5),
                                    'hf_std': round(hf_std, 5)})
        except Exception as e:
            logger.warning(f"DCT check failed: {e}")
            return DetectionResult('dct', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 4. CLONE / COPY-PASTE DETECTION
    # Are any regions visually duplicated within the document?
    # Common when a valid stamp or signature is copied from another cert.
    # ------------------------------------------------------------------ #

    def _check_clones(self, image: np.ndarray) -> DetectionResult:
        try:
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            sift  = cv2.SIFT_create(nfeatures=500)
            kps, descs = sift.detectAndCompute(gray, None)

            if descs is None or len(descs) < 10:
                return DetectionResult('clone', False, 0.0,
                                       "Not enough keypoints to check for clones.")

            bf      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(descs, descs, k=3)

            seen, clone_pairs = set(), []
            for group in matches:
                for m in group:
                    if m.queryIdx == m.trainIdx:
                        continue
                    pt1 = np.array(kps[m.queryIdx].pt)
                    pt2 = np.array(kps[m.trainIdx].pt)
                    if np.linalg.norm(pt1 - pt2) > 30 and m.distance < 50:
                        key = tuple(sorted([tuple(pt1.astype(int)), tuple(pt2.astype(int))]))
                        if key not in seen:
                            seen.add(key)
                            clone_pairs.append((pt1, pt2))

            count     = len(clone_pairs)
            threshold = 5
            looks_wrong = count >= threshold
            reason = (
                f"Found {count} pairs of visually identical but spatially separated regions — "
                f"strongly suggests a stamp, signature, or field was copied from another document."
                if looks_wrong else
                f"No suspicious duplicate regions found ({count} weak matches, below threshold of {threshold})."
            )

            return DetectionResult('clone', looks_wrong, round(min(count / 20, 1.0), 2),
                                   reason,
                                   {'clone_pair_count': count})
        except Exception as e:
            logger.warning(f"Clone check failed: {e}")
            return DetectionResult('clone', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 5. EDGE INCONSISTENCY
    # Do edges look natural throughout the document?
    # Pasted content introduces edges that don't exist at coarser scales —
    # the document's scan blur isn't applied uniformly.
    # ------------------------------------------------------------------ #

    def _check_edges(self, image: np.ndarray) -> DetectionResult:
        try:
            gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            fine   = sobel(gray)
            coarse = sobel(cv2.GaussianBlur(gray, (5, 5), 2))
            inconsistency = float(np.mean(np.abs(fine - coarse)))

            threshold = 0.08 * self._s
            looks_wrong = inconsistency > threshold
            reason = (
                f"Edge sharpness is inconsistent across the document (score {inconsistency:.4f} vs "
                f"threshold {threshold:.4f}) — some regions appear sharper than the surrounding scan, "
                f"suggesting composited content."
                if looks_wrong else
                "Edge profiles are consistent with a uniformly scanned document."
            )

            return DetectionResult('edge', looks_wrong, round(min(inconsistency / threshold, 1.0), 2),
                                   reason,
                                   {'inconsistency_score': round(inconsistency, 5)})
        except Exception as e:
            logger.warning(f"Edge check failed: {e}")
            return DetectionResult('edge', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 6. BACKGROUND LUMINANCE CONSISTENCY
    # Is the white paper background the same shade throughout?
    # Different lighting sources reveal the document was assembled
    # from more than one physical or digital source.
    # ------------------------------------------------------------------ #

    def _check_luminance(self, image: np.ndarray) -> DetectionResult:
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l   = lab[:, :, 0].astype(np.float32)

            _, bg_mask = cv2.threshold(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                200, 255, cv2.THRESH_BINARY
            )
            bg_frac = float(np.mean(bg_mask / 255.0))

            if bg_frac < 0.2:
                return DetectionResult('luminance', False, 0.0,
                                       "Not enough background area to evaluate luminance.")

            bg_l      = l[bg_mask > 0]
            lum_range = float(np.percentile(bg_l, 95) - np.percentile(bg_l, 5))
            threshold = 25.0 * self._s

            looks_wrong = lum_range > threshold
            reason = (
                f"Background luminance varies by {lum_range:.1f} units across the page "
                f"(expected < {threshold:.1f}) — the paper brightness is not uniform, "
                f"which can indicate the document was assembled from multiple scans."
                if looks_wrong else
                f"Background luminance is consistent (range {lum_range:.1f}) — single-source scan."
            )

            return DetectionResult('luminance', looks_wrong,
                                   round(min(lum_range / threshold, 1.0), 2),
                                   reason,
                                   {'luminance_range_90pct': round(lum_range, 2),
                                    'background_fraction': round(bg_frac, 3)})
        except Exception as e:
            logger.warning(f"Luminance check failed: {e}")
            return DetectionResult('luminance', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 7. TEXT FIELD INTEGRITY
    # Do individual text components have consistent ink and noise?
    # Altered fields (name, date, plot number) betray themselves through
    # localised ELA hotspots at the character level.
    # ------------------------------------------------------------------ #

    def _check_text_integrity(self, image: np.ndarray) -> DetectionResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            text_mask = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 4
            )
            text_mask = cv2.morphologyEx(
                text_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            )

            if np.mean(text_mask / 255.0) < 0.01:
                return DetectionResult('text_integrity', False, 0.0,
                                       "No text regions detected.")

            ela       = self._ela_map(image)
            ela_gray  = np.mean(ela, axis=2)
            num_labels, labels = cv2.connectedComponents(text_mask)

            component_elas = []
            for label_id in range(1, min(num_labels, 300)):
                mask = labels == label_id
                if np.sum(mask) < 20:
                    continue
                component_elas.append(float(np.mean(ela_gray[mask])))

            if not component_elas:
                return DetectionResult('text_integrity', False, 0.0,
                                       "Could not measure individual text components.")

            arr      = np.array(component_elas)
            ela_mean = float(np.mean(arr))
            ela_std  = float(np.std(arr))
            # Components whose ELA is significantly higher than the rest
            outliers = int(np.sum(arr > ela_mean + 2.0 * ela_std))

            threshold = 5
            looks_wrong = outliers > threshold
            reason = (
                f"{outliers} text components have compression signatures that differ from "
                f"the rest of the document's text — these characters or words may have been "
                f"typed or pasted in after the original document was created."
                if looks_wrong else
                f"All text components have consistent compression signatures ({outliers} outliers, "
                f"below threshold of {threshold})."
            )

            return DetectionResult('text_integrity', looks_wrong,
                                   round(min(outliers / 20, 1.0), 2),
                                   reason,
                                   {'text_component_count': num_labels,
                                    'ela_outlier_count': outliers,
                                    'ela_mean': round(ela_mean, 5),
                                    'ela_std': round(ela_std, 5)})
        except Exception as e:
            logger.warning(f"Text integrity check failed: {e}")
            return DetectionResult('text_integrity', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # 8. FREQUENCY DOMAIN ANOMALIES
    # Does the document's frequency spectrum look natural?
    # Double-compression, upscaling, or tiling leaves spectral fingerprints
    # that a genuine single-scan document would not have.
    # ------------------------------------------------------------------ #

    def _check_frequency(self, image: np.ndarray) -> DetectionResult:
        try:
            gray      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            fshift    = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = np.log1p(np.abs(fshift))

            h, w   = magnitude.shape
            cy, cx = h // 2, w // 2

            # Radial energy profile
            y_idx, x_idx = np.mgrid[-cy:h-cy, -cx:w-cx]
            radii    = np.sqrt(x_idx**2 + y_idx**2).astype(int)
            max_r    = min(cy, cx)
            profile  = np.array([
                np.mean(magnitude[radii == r]) if np.any(radii == r) else 0.0
                for r in range(max_r)
            ])

            smooth    = ndimage.uniform_filter1d(profile, size=5)
            residual  = np.abs(profile - smooth)
            anomaly   = float(np.mean(residual[5:]) / (np.mean(smooth[5:]) + 1e-8))

            # Grid artifacts from tiling/copy-paste
            grid_ratio = float(
                (np.mean(magnitude[cy-2:cy+2, :]) + np.mean(magnitude[:, cx-2:cx+2]))
                / (np.mean(magnitude) + 1e-8)
            )

            threshold_anomaly = 0.07 * self._s
            looks_wrong = anomaly > threshold_anomaly or grid_ratio > 4.0
            reason = (
                (f"Frequency spectrum shows anomalous spikes (anomaly score {anomaly:.4f}) — "
                 f"consistent with double-compression or content upscaled from a lower resolution."
                 if anomaly > threshold_anomaly else
                 f"Strong grid pattern in frequency domain (ratio {grid_ratio:.2f}) — "
                 f"suggests tiled or periodically repeated content (copied stamp or watermark).")
                if looks_wrong else
                "Frequency spectrum matches expected profile of a single-scan document."
            )

            return DetectionResult('frequency', looks_wrong,
                                   round(min(max(anomaly / threshold_anomaly,
                                                 grid_ratio / 4.0), 1.0), 2),
                                   reason,
                                   {'anomaly_score': round(anomaly, 5),
                                    'grid_ratio': round(grid_ratio, 4)})
        except Exception as e:
            logger.warning(f"Frequency check failed: {e}")
            return DetectionResult('frequency', False, 0.0, f"Check could not run: {e}")

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Run all visual checks and answer: does this document look forged?

        Args:
            image: RGB numpy array

        Returns:
            {
              'looks_forged': bool,
              'verdict':      'LOOKS AUTHENTIC' | 'LOOKS SUSPICIOUS' | 'LOOKS FORGED',
              'summary':      str,   # plain-English explanation
              'flags':        list,  # which checks flagged and why
              'checks':       dict,  # per-method detail
            }
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return {
                'looks_forged': False,
                'verdict': 'ERROR',
                'summary': 'Invalid image — could not analyse.',
                'flags': [],
                'checks': {}
            }

        # Cap resolution to keep runtime reasonable
        h, w = image.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        results: List[DetectionResult] = [
            self._check_ela(image),
            self._check_noise(image),
            self._check_dct(image),
            self._check_clones(image),
            self._check_edges(image),
            self._check_luminance(image),
            self._check_text_integrity(image),
            self._check_frequency(image),
        ]

        flags = [r for r in results if r.looks_wrong and r.confidence > 0]
        flag_count = len(flags)

        if flag_count >= 4:
            verdict      = 'LOOKS FORGED'
            looks_forged = True
            summary      = (f"{flag_count} of 8 visual checks failed. "
                            f"The document shows multiple independent signs of manipulation.")
        elif flag_count >= 2:
            verdict      = 'LOOKS SUSPICIOUS'
            looks_forged = True
            summary      = (f"{flag_count} of 8 visual checks raised concerns. "
                            f"The document warrants closer inspection.")
        else:
            verdict      = 'LOOKS AUTHENTIC'
            looks_forged = False
            summary      = (f"Only {flag_count} of 8 checks raised minor concerns. "
                            f"No strong visual evidence of manipulation.")

        return {
            'looks_forged': looks_forged,
            'verdict': verdict,
            'summary': summary,
            'sensitivity': self.sensitivity,
            'flags': [
                {'check': r.method, 'reason': r.reason, 'confidence': r.confidence}
                for r in flags
            ],
            'checks': {
                r.method: {
                    'looks_wrong': r.looks_wrong,
                    'confidence': r.confidence,
                    'reason': r.reason,
                    'details': r.details,
                }
                for r in results
            }
        }