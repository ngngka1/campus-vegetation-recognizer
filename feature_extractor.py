import cv2
import numpy as np


def _make_hog_descriptor(img_size: int) -> cv2.HOGDescriptor:
    if img_size < 16:
        raise ValueError("img_size must be >= 16 for HOG feature extraction.")
    # Keep window aligned to HOG cell grid.
    win = (img_size // 8) * 8
    if win < 16:
        win = 16
    return cv2.HOGDescriptor(
        _winSize=(win, win),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

def extract_hog_features(images: list[np.ndarray], img_size: int) -> np.ndarray:
    hog = _make_hog_descriptor(img_size)
    feats = []
    for img in images:
        if img.shape[0] != img_size or img.shape[1] != img_size:
            raise ValueError(
                "All images must be resized to img_size before HOG extraction."
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vec = hog.compute(gray)
        feats.append(vec.reshape(-1))
    return np.asarray(feats, dtype=np.float32)


def extract_hsv_hist_features(images: list[np.ndarray]) -> np.ndarray:
    feats = []
    bins = (12, 8, 8)
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.asarray(feats, dtype=np.float32)


def extract_color_moment_features(images: list[np.ndarray]) -> np.ndarray:
    feats = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv)
        vec: list[float] = []
        for ch in channels:
            arr = ch.astype(np.float32).ravel()
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            centered = arr - mean
            # Use stable cube-root on third central moment; avoids NaN for negative moments.
            third_moment = float(np.mean(centered**3))
            skew = float(np.cbrt(third_moment))
            vec.extend([mean, std, skew])
        feats.append(vec)
    return np.asarray(feats, dtype=np.float32)


def extract_lbp_features(images: list[np.ndarray]) -> np.ndarray:
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center = gray[1:-1, 1:-1]
        codes = np.zeros(center.shape, dtype=np.uint8)

        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
        for bit, (dy, dx) in enumerate(offsets):
            neigh = gray[
                1 + dy : gray.shape[0] - 1 + dy, 1 + dx : gray.shape[1] - 1 + dx
            ]
            codes |= (neigh >= center).astype(np.uint8) << bit

        hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
        hist /= max(1.0, hist.sum())
        feats.append(hist)
    return np.asarray(feats, dtype=np.float32)


def extract_glcm_features(images: list[np.ndarray]) -> np.ndarray:
    distances = (1, 2, 4)
    angles = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    levels = 16
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        quant = (gray // (256 // levels)).astype(np.int32)
        h, w = quant.shape
        vals: list[float] = []
        for d in distances:
            for a in angles:
                dx = int(round(np.cos(a) * d))
                dy = int(round(np.sin(a) * d))
                x0 = max(0, -dx)
                x1 = min(w, w - dx)
                y0 = max(0, -dy)
                y1 = min(h, h - dy)
                if x0 >= x1 or y0 >= y1:
                    vals.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                p = quant[y0:y1, x0:x1].ravel()
                q = quant[y0 + dy : y1 + dy, x0 + dx : x1 + dx].ravel()
                glcm = np.zeros((levels, levels), dtype=np.float64)
                np.add.at(glcm, (p, q), 1.0)
                s = glcm.sum()
                if s <= 0:
                    vals.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                glcm /= s
                i = np.arange(levels, dtype=np.float64)[:, None]
                j = np.arange(levels, dtype=np.float64)[None, :]
                contrast = float(np.sum(((i - j) ** 2) * glcm))
                homogeneity = float(np.sum(glcm / (1.0 + np.abs(i - j))))
                energy = float(np.sum(glcm * glcm))
                mu_i = float(np.sum(i * glcm))
                mu_j = float(np.sum(j * glcm))
                sigma_i = float(np.sqrt(np.sum(((i - mu_i) ** 2) * glcm)))
                sigma_j = float(np.sqrt(np.sum(((j - mu_j) ** 2) * glcm)))
                if sigma_i <= 1e-12 or sigma_j <= 1e-12:
                    correlation = 0.0
                else:
                    correlation = float(
                        np.sum(((i - mu_i) * (j - mu_j) * glcm)) / (sigma_i * sigma_j)
                    )
                vals.extend([contrast, homogeneity, energy, correlation])
        feats.append(vals)
    return np.asarray(feats, dtype=np.float32)

def _aggregate_local_descriptors(
    images: list[np.ndarray],
    *,
    descriptor_name: str,
    max_keypoints: int = 200,
) -> np.ndarray:
    if descriptor_name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable in this OpenCV build.")
        extractor = cv2.SIFT_create(nfeatures=max_keypoints)
    elif descriptor_name == "surf":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(
            cv2.xfeatures2d, "SURF_create"
        ):
            raise RuntimeError("SURF is unavailable (requires opencv-contrib).")
        extractor = cv2.xfeatures2d.SURF_create(200)
    else:
        raise ValueError(f"Unknown descriptor_name: {descriptor_name}")

    feats = []
    dim = 128 if descriptor_name == "sift" else 64
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _kp, desc = extractor.detectAndCompute(gray, None)
        if desc is None or desc.shape[0] == 0:
            vec = np.zeros((dim * 2 + 1,), dtype=np.float32)
        else:
            desc = desc.astype(np.float32)
            mean = desc.mean(axis=0)
            std = desc.std(axis=0)
            count = np.array([float(desc.shape[0])], dtype=np.float32)
            vec = np.concatenate([mean, std, count], axis=0).astype(np.float32)
        feats.append(vec)
    return np.asarray(feats, dtype=np.float32)  
  
def extract_sift_features(images: list[np.ndarray]) -> np.ndarray:
    return _aggregate_local_descriptors(images, descriptor_name="sift")


def extract_surf_features(images: list[np.ndarray]) -> np.ndarray:
    return _aggregate_local_descriptors(images, descriptor_name="surf")