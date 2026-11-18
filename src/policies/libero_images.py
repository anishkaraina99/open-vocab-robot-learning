"""Shared image orientation for LIBERO training, perception, and live inference."""
import numpy as np

IMAGE_TRANSFORM = 'rot180:matches_language_baseline'


def policy_image(image):
    """Rotate a raw stored/simulator RGB frame exactly once into policy orientation."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 HWC RGB image')
    return np.rot90(image, k=2).copy()
