from pathlib import Path

import cv2
import numpy as np
from pydicom import dcmread
from scipy import ndimage
from skimage import exposure


def load_dataset(path):
    ds = dcmread(Path(path).joinpath("DICOMDIR"))

    return ds


def apply_simple_denoise(img, denoise_filter=ndimage.median_filter, kernel_size=3):
    new_img = denoise_filter(img.pixel_array, kernel_size)

    return new_img


def apply_non_local_means(img, strength=10, kernel=7, window_search=21):
    dcm_sample = img.pixel_array * 128
    dcm_sample = exposure.equalize_adapthist(dcm_sample)
    dcm_sample *= 255

    img_uint8 = np.uint8(dcm_sample)

    return cv2.fastNlMeansDenoising(img_uint8, None, strength, kernel, window_search)
