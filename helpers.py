from pathlib import Path
from pydicom import dcmread
from scipy import ndimage


def load_dataset(path):
    ds = dcmread(Path(path).joinpath("DICOMDIR"))

    return ds


def apply_denoise(img, kernel_size=3):
    new_img = ndimage.median_filter(img.pixel_array, kernel_size)

    return new_img
