from pathlib import Path
from pydicom import dcmread
from scipy import ndimage


def load_dataset(path):
    ds = dcmread(Path(path).joinpath("DICOMDIR"))

    return ds


def apply_denoise(img, denoise_filter=ndimage.median_filter, kernel_size=3):
    new_img = denoise_filter(img.pixel_array, kernel_size)

    return new_img
