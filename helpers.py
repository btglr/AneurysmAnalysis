from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
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


# d: Diameter of each pixel neighborhood.
# sigmaColor: Value of \sigma in the color space.
# The greater the value, the colors farther to each other
# will start to get mixed.
# sigmaColor: Value of \sigma in the coordinate space.
# The greater its value, the more further pixels will mix together,
# given that their colors lie within the sigmaColor range.
def apply_bilateral_filtering(img, d=15, sigmacolor=75, sigmacoordinate=75):
    dcm_sample = img.pixel_array * 128
    dcm_sample = exposure.equalize_adapthist(dcm_sample)
    dcm_sample *= 255

    img_uint8 = np.uint8(dcm_sample)

    return cv2.bilateralFilter(img_uint8, d, sigmacolor, sigmacoordinate)


def plot_slider(images):
    img = plt.imshow(images[0], cmap=plt.cm.bone)

    def update(val):
        img.set_data(images[int(val)])

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axamp, 'Images', 0, images.__len__() - 1, valinit=0, valstep=1, valfmt="%i")
    samp.on_changed(update)

    plt.show()
