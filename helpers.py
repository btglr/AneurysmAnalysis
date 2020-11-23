from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from pydicom import dcmread
from scipy import ndimage
from skimage.morphology import flood_fill
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_bilateral
from skimage.segmentation import random_walker

is_key_held = False
fig = None
default_title = "Hold CTRL to make a selection"
ls = []
images_drawn = []
current_image_slider = 0
median_images = []


def load_dataset(path):
    ds = dcmread(Path(path).joinpath("DICOMDIR"))

    return ds


def apply_simple_denoise(img, denoise_filter=ndimage.median_filter, kernel_size=3):
    new_img = denoise_filter(img.pixel_array, kernel_size)

    return new_img


def apply_non_local_means(img, kernel=5, window_search=13):
    # Retain original data type
    orig_dtype = img.pixel_array.dtype

    # Convert from [0; max] to [0; 1] as it is required by denoise_nl_means
    upper_bound = np.max(img.pixel_array)
    img_as_float = img.pixel_array / upper_bound

    sigma_est = np.mean(estimate_sigma(img_as_float, multichannel=False))

    new_img = denoise_nl_means(img_as_float, h=sigma_est, fast_mode=True, patch_size=kernel,
                               patch_distance=window_search)

    # Convert back to [0; max]
    new_img *= upper_bound

    return new_img.astype(orig_dtype)


# d: Diameter of each pixel neighborhood.
# sigmaColor: Value of \sigma in the color space.
# The greater the value, the colors farther to each other
# will start to get mixed.
# sigmaColor: Value of \sigma in the coordinate space.
# The greater its value, the more further pixels will mix together,
# given that their colors lie within the sigmaColor range.
def apply_bilateral_filtering(img, d=15, sigmacolor=75, sigmacoordinate=75):
    # Retain original data type
    orig_dtype = img.pixel_array.dtype

    # Convert from [0; max] to [0; 1] as it is required by denoise_nl_means
    upper_bound = np.max(img.pixel_array)
    img_as_float = img.pixel_array / upper_bound

    new_img = denoise_bilateral(img_as_float, win_size=d, sigma_color=sigmacolor, sigma_spatial=sigmacoordinate)

    # Convert back to [0; max]
    new_img *= upper_bound

    return new_img.astype(orig_dtype)


def plot_slider(images, label=""):
    img = plt.imshow(images[0], cmap=plt.cm.bone)

    def update(val):
        img.set_data(images[int(val)])

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axamp, 'Images', 0, images.__len__() - 1, valinit=0, valstep=1, valfmt="%i")
    samp.on_changed(update)

    plt.xlabel(label)
    plt.show()


def subplots_slider(images, zoom=2.0, click_handler=None):
    global images_drawn
    images_drawn = images
    height, width = images_drawn[0][1][0].shape
    nb_image_sets = len(images_drawn)

    nrows = int(np.ceil(np.sqrt(nb_image_sets)))
    ncols = nb_image_sets // nrows

    # If we only have 3 images, add 2 columns
    if nrows == 2 and ncols == 1 and nb_image_sets == 3:
        ncols = 2

    # print(nrows, ncols)
    # print(height, width)

    global fig
    fig = plt.figure(figsize=((width * ncols * zoom) / 100, (height * nrows * zoom) / 100), dpi=100)
    fig.suptitle(default_title, fontsize=16)

    # print(fig.get_size_inches() * fig.dpi)

    global ls

    for k in range(nb_image_sets):
        if images_drawn[k][2]['type'] == 'median_filter':
            global median_images
            median_images = images_drawn[k][1]

        ax = fig.add_subplot(nrows, ncols, k + 1)
        image = ax.imshow(images_drawn[k][1][0], cmap=plt.cm.gray, aspect='auto')
        ls.append(image)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(images_drawn[k][0])

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    sframe = Slider(axamp, 'Image', 0, len(images_drawn[0][1]) - 1, valinit=0, valstep=1, valfmt="%i")

    def update(val):
        global current_image_slider
        val = int(val)
        current_image_slider = val

        for k, l in enumerate(ls):
            l.set_data(images_drawn[k][1][val])

    sframe.on_changed(update)
    # fig.subplots_adjust(wspace=0, hspace=0)

    if click_handler is not None:
        fig.canvas.mpl_connect("key_press_event", key_press_handler)
        fig.canvas.mpl_connect("key_release_event", key_release_handler)
        fig.canvas.mpl_connect("button_press_event", click_handler)

    plt.show()


def apply_threshold(image):
    # thresh = threshold_mean(image)
    return image > 300


def key_press_handler(event):
    if event.key == 'control':
        fig.suptitle("Left-click on an area to select a position", fontsize=16)
        fig.canvas.draw()

        global is_key_held
        is_key_held = True


def key_release_handler(event):
    if event.key == 'control':
        fig.suptitle(default_title, fontsize=16)
        fig.canvas.draw()

        global is_key_held
        is_key_held = False


def select_region(event):
    global is_key_held
    global ls

    if is_key_held and event.button == 1:
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))

        x = int(event.xdata)
        y = int(event.ydata)

        print("Apply flood fill at coordinates: ({}, {})".format(x, y))

        for k, l in enumerate(ls):
            params = images_drawn[k][2]

            if params['type'] == 'flood_fill':
                tol = params['tolerance']

                for i in range(len(median_images)):
                    images_drawn[k][1][i] = apply_flood_fill(median_images[i],
                                                             (x, y),
                                                             tolerance=tol)

                l.set_data(images_drawn[k][1][current_image_slider])

        fig.canvas.draw()


def apply_random_walker(image):
    upper_bound = np.max(image)
    img_as_float = image / upper_bound
    img_as_float *= 2
    img_as_float -= 1

    # print(img_as_float.max(), img_as_float.min())
    # print(image.min(), image.max())

    markers = np.zeros(img_as_float.shape, dtype=np.uint)
    markers[img_as_float < -0.90] = 1
    markers[img_as_float > 0.90] = 2

    return random_walker(img_as_float, markers, beta=10, mode='bf')


def apply_flood_fill(image, starting_coordinates, tolerance):
    upper_bound = np.max(image)
    img_as_float = image / upper_bound
    fill = flood_fill(img_as_float, starting_coordinates, 1.0, tolerance=tolerance)

    # Zero all values that haven't been replaced by the flood fill, effectively turning the image into a binary image
    fill[fill < 1.0] = 0

    return fill
