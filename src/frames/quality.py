import cv2
import numpy as np
from PIL import Image
import imagehash


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def perceptual_hash(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def quality_score(image):
    sharpness = blur_score(image)
    brightness = brightness_score(image)

    sharpness_score = min(sharpness / 500.0, 1.0)

    if brightness < 20 or brightness > 235:
        exposure_score = 0.0
    else:
        exposure_score = 1.0

    return 0.7 * sharpness_score + 0.3 * exposure_score