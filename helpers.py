import copy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.widgets import Slider, TextBox, Button
from pydicom import dcmread
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import flood
from skimage.morphology import skeletonize_3d
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_bilateral
from skimage.segmentation import random_walker

import globals


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
    img = plt.imshow(images[0], cmap=plt.cm.gray)

    def update(val):
        img.set_data(images[int(val)])

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axamp, 'Images', 0, images.__len__() - 1, valinit=0, valstep=1, valfmt="%i")
    samp.on_changed(update)

    plt.xlabel(label)
    plt.show()


def subplots_slider(images, zoom=2.0, click_handler=None):
    globals.images_drawn = images
    globals.ls = []
    height, width = globals.images_drawn[0][1][0].shape
    nb_image_sets = len(globals.images_drawn)

    ncols = nb_image_sets if nb_image_sets <= 6 else 6
    nrows = int(np.ceil(nb_image_sets / float(ncols)))

    # print(nrows, ncols)
    # print(height, width)

    globals.fig = plt.figure(figsize=((width * ncols * zoom) / 100, (height * nrows * zoom) / 100), dpi=100)
    globals.fig.suptitle(globals.default_title, fontsize=16)

    # print(fig.get_size_inches() * fig.dpi)

    for k in range(nb_image_sets):
        ax = globals.fig.add_subplot(nrows, ncols, k + 1)
        vmin = globals.images_drawn[k][1][0].min()
        vmax = globals.images_drawn[k][1][0].max()

        if vmin == vmax:
            vmin = 0
            vmax = 1

        image = ax.imshow(globals.images_drawn[k][1][0], cmap=plt.cm.bone, aspect='auto', vmin=vmin, vmax=vmax)
        globals.ls.append(image)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(globals.images_drawn[k][0])

    ax_image_slider = plt.axes([0.25, .03, 0.50, 0.02])
    ax_flood_fill_tolerance_textbox = plt.axes([0.06, 0.95, 0.05, 0.04])
    ax_seed_tolerance_textbox = plt.axes([0.06, 0.90, 0.05, 0.04])
    ax_resize_factor_textbox = plt.axes([0.06, 0.85, 0.05, 0.04])
    ax_recalculate_button = plt.axes([0.01, 0.80, 0.10, 0.04])
    ax_save_button = plt.axes([0.01, 0.75, 0.10, 0.04])
    ax_reset_button = plt.axes([0.01, 0.01, 0.10, 0.04])

    image_slider = Slider(ax_image_slider, 'Image', 0, len(globals.images_drawn[0][1]) - 1, valinit=0, valstep=1,
                          valfmt="%i")
    flood_fill_tolerance_textbox = TextBox(ax_flood_fill_tolerance_textbox, 'Flood Fill Tolerance',
                                           initial=str(globals.flood_fill_tolerance))
    seed_tolerance_textbox = TextBox(ax_seed_tolerance_textbox, 'Seed Tolerance',
                                     initial=str(globals.seed_tolerance))
    resize_factor_textbox = TextBox(ax_resize_factor_textbox, 'Resize Skel. Factor',
                                    initial=str(globals.skeleton_factor))
    recalculate_button = Button(ax_recalculate_button, 'Recalculate segmentation', color='0.85', hovercolor='0.95')
    save_button = Button(ax_save_button, 'Save mask and skeleton', color='0.85', hovercolor='0.95')
    reset_button = Button(ax_reset_button, 'Reset segmentations', color='0.85', hovercolor='red')

    result_element = [image_set for image_set in globals.images_drawn if image_set[0] == 'Result'][0]
    skeleton_element = [image_set for image_set in globals.images_drawn if image_set[0] == 'Skeleton'][0]
    index_result = globals.images_drawn.index(result_element)
    index_skeleton = globals.images_drawn.index(skeleton_element)

    def update(val):
        val = int(val)
        globals.current_image_slider = val

        for k, l in enumerate(globals.ls):
            l.set_data(globals.images_drawn[k][1][val])

    def update_flood_fill_tolerance(text):
        if globals.flood_fill_tolerance != float(text):
            globals.flood_fill_tolerance = float(text)

    def update_seed_tolerance(text):
        if globals.seed_tolerance != float(text):
            globals.seed_tolerance = float(text)

    def update_resize_factor(text):
        if globals.skeleton_factor != int(text):
            globals.skeleton_factor = int(text)

    def save_result(event):
        original_images = copy.deepcopy(globals.images)

        for i, elem in enumerate(globals.images_drawn[index_result][1]):
            image_elem, p = original_images[i]
            filename = p.name
            study_folder = p.parent.parent

            image_elem.PixelData = globals.images_drawn[index_result][1][i].astype('uint16').tobytes()

            result_folder = Path(globals.dataset_path).joinpath(study_folder).joinpath('result')
            result_folder.mkdir(parents=True, exist_ok=True)

            filepath = result_folder.joinpath(filename)
            image_elem.save_as(filepath)

            image_elem.PixelData = globals.images_drawn[index_skeleton][1][i].astype('uint16').tobytes()
            image_elem.Rows, image_elem.Columns = globals.images_drawn[index_skeleton][1][0].shape

            skeleton_folder = Path(globals.dataset_path).joinpath(study_folder).joinpath('skeleton')
            skeleton_folder.mkdir(parents=True, exist_ok=True)

            filepath = skeleton_folder.joinpath(filename)
            image_elem.save_as(filepath)

    def recalculate_segmentation(event):
        if globals.selected_segmentation != -1:
            apply_flood_fill_subplots(globals.list_segmentations[globals.selected_segmentation][1],
                                      starting_image=globals.list_segmentations[globals.selected_segmentation][0],
                                      segmentation_index=globals.selected_segmentation)
        else:
            apply_flood_fill_subplots(globals.starting_coordinates, starting_image=globals.starting_image)

    def reset_segmentations(_):
        globals.segmentations_masks.clear()
        globals.segmentations_results.clear()

        for index, image_set in enumerate(globals.ls):
            params = globals.images_drawn[index][2]

            if params['type'] == 'flood_fill' or params['type'] == 'result' or params['type'] == 'skeleton':
                globals.images_drawn[index][1] = [np.zeros(globals.images_drawn[index][1][0].shape)] * len(
                    globals.images_drawn[index][1])
                image_set.set_data(globals.images_drawn[index][1][globals.current_image_slider])

    image_slider.on_changed(update)
    flood_fill_tolerance_textbox.on_submit(update_flood_fill_tolerance)
    seed_tolerance_textbox.on_submit(update_seed_tolerance)
    resize_factor_textbox.on_submit(update_resize_factor)
    save_button.on_clicked(save_result)

    recalculate_button.on_clicked(recalculate_segmentation)
    reset_button.on_clicked(reset_segmentations)
    # fig.subplots_adjust(wspace=0, hspace=0)

    if click_handler is not None:
        globals.fig.canvas.mpl_connect("key_press_event", key_press_handler)
        globals.fig.canvas.mpl_connect("key_release_event", key_release_handler)
        globals.fig.canvas.mpl_connect("button_press_event", click_handler)

    plt.show()


def apply_threshold(image):
    # thresh = threshold_mean(image)
    return image > 300


def key_press_handler(event):
    if event.key == 'control':
        globals.fig.suptitle("Left-click on an area to select a position", fontsize=16)
        globals.fig.canvas.draw()

        globals.is_key_held = True


def key_release_handler(event):
    if event.key == 'control':
        globals.fig.suptitle(globals.default_title, fontsize=16)
        globals.fig.canvas.draw()

        globals.is_key_held = False


def select_region(event):
    if globals.is_key_held and event.button == 1:
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))

        x = int(event.xdata)
        y = int(event.ydata)

        globals.starting_coordinates = (y, x)
        apply_flood_fill_subplots(globals.starting_coordinates, starting_image=globals.current_image_slider)
        globals.fig.canvas.draw()


def apply_flood_fill_subplots(coordinates, starting_image=None, flood_fill_tolerance=None, segmentation_index=-1,
                              seed_tolerance=None):
    changing_segmentation = False

    for index, image_set in enumerate(globals.ls):
        params = globals.images_drawn[index][2]

        if params['type'] == 'flood_fill':
            if flood_fill_tolerance is None:
                flood_fill_tolerance = globals.flood_fill_tolerance

            if seed_tolerance is None:
                seed_tolerance = globals.seed_tolerance

            params['flood_fill_tolerance'] = flood_fill_tolerance
            params['seed_tolerance'] = seed_tolerance

            # Coordinates are height, width instead of width, height in numpy
            # We therefore apply the flood fill to the coordinates (y, x)
            tmp_masks, tmp_results = evolutive_flood_fill(globals.median_images, flood_fill_tolerance,
                                                          coordinates,
                                                          starting_image=starting_image)

            if segmentation_index != -1:
                globals.segmentations_masks[segmentation_index] = [tmp_masks, params, 'Active']
                globals.segmentations_results[segmentation_index] = [tmp_results, params, 'Active']
                changing_segmentation = True
            else:
                globals.segmentations_masks.append([tmp_masks, params, 'Active'])
                globals.segmentations_results.append([tmp_results, params, 'Active'])
                segmentation_index = len(globals.segmentations_masks) - 1

            active_segmentations = [segmentation for segmentation in globals.segmentations_masks if
                                    segmentation[2] == 'Active']
            local_index = len(active_segmentations) - 1

            if changing_segmentation:
                print(f'Changed segmentation {segmentation_index}')
            else:
                print(f'Added segmentation {segmentation_index}')

                ax_delete_button = plt.axes([0.10, 0.65 - (local_index * 0.06), 0.01, 0.04],
                                            label=f'Delete {segmentation_index}')
                ax_select_button = plt.axes([0.085, 0.65 - (local_index * 0.06), 0.01, 0.04],
                                            label=f'Select {segmentation_index}')

                seg_text = plt.text(0.003, 0.646 - (local_index * 0.06),
                                    "Seg: {}, Coords: {}\nFlood: {}, Seed: {}".format(segmentation_index,
                                                                                      coordinates,
                                                                                      params['flood_fill_tolerance'],
                                                                                      params['seed_tolerance']),
                                    transform=plt.gcf().transFigure)

                delete_button = Button(ax_delete_button, 'X', color='0.85', hovercolor='red')
                select_button = Button(ax_select_button, 'S', color='0.85', hovercolor='0.95')
                segmentation_index_copy = segmentation_index
                current_axis = plt.gca()

                def delete_segmentation(_):
                    print(f'Removing segmentation {segmentation_index_copy}')

                    delete_button.active = False
                    select_button.active = False

                    del current_axis.texts[0]
                    ax_delete_button.remove()
                    ax_select_button.remove()

                    globals.segmentations_masks[segmentation_index_copy][2] = 'Inactive'
                    globals.segmentations_results[segmentation_index_copy][2] = 'Inactive'

                    redraw_segmentations()
                    plt.draw()

                def select_segmentation(_):
                    print(f'Selected segmentation {segmentation_index_copy}')
                    globals.selected_segmentation = segmentation_index_copy

                delete_button.on_clicked(delete_segmentation)
                select_button.on_clicked(select_segmentation)

                globals.list_segmentations.append(
                    [starting_image, coordinates, ax_delete_button, delete_button, select_button, seg_text])

    redraw_segmentations()


def redraw_segmentations():
    def merge_arrays(a, b):
        result = a.copy()
        zero_indexes = (a == 0)
        result[zero_indexes] = b[zero_indexes]

        return result

    for index, image_set in enumerate(globals.ls):
        params = globals.images_drawn[index][2]

        if params['type'] == 'flood_fill':
            active_segmentations = [segmentation for segmentation in globals.segmentations_masks if
                                    segmentation[2] == 'Active']

            if len(active_segmentations) == 0:
                globals.images_drawn[index][1] = [np.zeros(
                    globals.images_drawn[index][1][0].shape)] * len(
                    globals.images_drawn[index][1])
                globals.results = [np.zeros(globals.images_drawn[index][1][0].shape)] * len(
                    globals.images_drawn[index][1])

            globals.images_drawn[index][1] = None
            globals.results = None

            for segmentation_masks_tuple, segmentation_results_tuple in zip(globals.segmentations_masks,
                                                                            globals.segmentations_results):

                if segmentation_masks_tuple[2] == 'Active':
                    segmentation_masks, segmentation_results = segmentation_masks_tuple[0], segmentation_results_tuple[
                        0]

                    if globals.images_drawn[index][1] is None and globals.results is None:
                        globals.images_drawn[index][1] = segmentation_masks
                        globals.results = segmentation_results
                    else:
                        globals.images_drawn[index][1] = [merge_arrays(drawn, new) for drawn, new in
                                                          zip(globals.images_drawn[index][1],
                                                              segmentation_masks)]
                        globals.results = [merge_arrays(drawn, new) for drawn, new in
                                           zip(globals.results, segmentation_results)]

            # If None, there are no images so we set a blank image
            if globals.images_drawn[index][1] is None:
                globals.images_drawn[index][1] = [np.zeros(globals.median_images[0].shape)] * len(globals.median_images)
                globals.results = [np.zeros(globals.median_images[0].shape)] * len(globals.median_images)

            image_set.set_data(globals.images_drawn[index][1][globals.current_image_slider])

        elif params['type'] == 'result':
            globals.images_drawn[index][1] = globals.results
            image_set.set_data(globals.images_drawn[index][1][globals.current_image_slider])

        elif params['type'] == 'skeleton':
            mask = [image_set for image_set in globals.images_drawn if image_set[0] == 'Mask'][0]
            globals.images_drawn[index][1] = resize_and_skeleton_3d(mask[1], globals.skeleton_factor)
            image_set.set_data(globals.images_drawn[index][1][globals.current_image_slider])


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
    mask = flood(img_as_float, starting_coordinates, tolerance=tolerance)
    mask = mask.astype('uint16')

    return mask


def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)


def evolve_fill(images, begin, end, mask_starting_image, gray_at_starting_coordinates, flood_fill_tolerance,
                starting_coordinates):
    step = 1
    mask = [np.zeros(mask_starting_image.shape)] * len(images)
    result = [np.zeros(mask_starting_image.shape)] * len(images)

    gray_at_coordinates = gray_at_starting_coordinates

    if begin > end:
        step = -1
        end -= 1
    else:
        begin += 1

    print("Beginning evolution of flood fill from image {}".format(begin))
    print("Looping from {} to {}\n".format(begin, end))
    mask[begin - step] = mask_starting_image

    # print(begin, end, step)

    for image_number in range(begin, end, step):
        print("Image {}".format(image_number))

        selected_gray = images[image_number]
        atol = int(globals.seed_tolerance * gray_at_coordinates)

        print("  Searching for grays within ± {} of the gray value {}".format(atol, gray_at_coordinates))

        # Find a new starting point
        close_values = np.where(np.isclose(selected_gray * mask[image_number - step], gray_at_coordinates, atol=atol))

        # Combine the two 1D arrays so we get an array of (y, x) coordinates
        combined = np.column_stack(close_values)

        if combined.shape[0] == 0:
            print("  Found no new starting point with the given tolerance\n")
            continue

        distances = cdist(np.array([starting_coordinates]), combined)
        values = [int(selected_gray[combined[i][0], combined[i][1]]) - int(gray_at_coordinates) for i in
                  range(combined.shape[0])]

        # Sort by how close to the starting gray value the new values are, then by their distance to the starting
        # coordinates
        idx_sort = np.lexsort((distances[0], np.abs(values)))
        new_coordinates = (combined[idx_sort[0]][0], combined[idx_sort[0]][1])

        # for i in range(5 if len(idx_sort) > 5 else len(idx_sort)):
        #     coords = (combined[idx_sort[i]][0], combined[idx_sort[i]][1])
        #     d = distances[0][idx_sort[i]]
        #     print(coords, d, selected_gray[coords])

        gray_at_coordinates = selected_gray[new_coordinates]

        print("  New starting coordinates: ({}, {}) | Value: {}\n".format(new_coordinates[1], new_coordinates[0],
                                                                          gray_at_coordinates))

        mask[image_number] = apply_flood_fill(selected_gray, new_coordinates, flood_fill_tolerance)
        selected_fill = mask[image_number]
        selected_masked = selected_gray * selected_fill
        result[image_number] = selected_masked

    return mask, result


def evolutive_flood_fill(images, flood_fill_tolerance, starting_coordinates,
                         starting_image=None):
    # Select image
    image_number = starting_image if starting_image is not None else globals.current_image_slider
    globals.starting_image = image_number
    globals.starting_coordinates = starting_coordinates

    # Select the image number from each set
    selected_gray = images[image_number]
    mask_starting_image = apply_flood_fill(selected_gray, starting_coordinates, flood_fill_tolerance)

    # Multiply the gray image with the mask
    selected_masked = selected_gray * mask_starting_image

    # Get the gray value at the starting coordinates
    gray_at_starting_coordinates = images[image_number][starting_coordinates]

    print("Apply flood fill at coordinates: ({}, {})".format(starting_coordinates[1], starting_coordinates[0]))
    print("Gray value for image {}: {}\n".format(image_number, gray_at_starting_coordinates))

    if image_number == 0:
        mask, result = evolve_fill(images, 1, len(images), mask_starting_image, gray_at_starting_coordinates,
                                   flood_fill_tolerance, starting_coordinates)
        mask = [mask_starting_image] + mask[1:len(images)]
        result = [selected_masked] + result[1:len(images)]
    elif image_number == len(images) - 1:
        mask, result = evolve_fill(images, 0, len(images) - 1, mask_starting_image, gray_at_starting_coordinates,
                                   flood_fill_tolerance, starting_coordinates)
        mask = mask[0:len(images) - 1] + [mask_starting_image]
        result = result[0:len(images) - 1] + selected_masked
    else:
        mask_upper, result_upper = evolve_fill(images, image_number, len(images), mask_starting_image,
                                               gray_at_starting_coordinates, flood_fill_tolerance,
                                               starting_coordinates)
        mask_lower, result_lower = evolve_fill(images, image_number - 1, 0, mask_starting_image,
                                               gray_at_starting_coordinates, flood_fill_tolerance,
                                               starting_coordinates)

        mask = mask_lower[0:image_number] + [mask_starting_image] + mask_upper[image_number + 1:len(images)]
        result = result_lower[0:image_number] + [selected_masked] + result_upper[image_number + 1:len(images)]

    return mask, result


def resize_and_skeleton_3d(mask, factor, dilation_kernel=None):
    height, width = mask[0].shape
    dsize = (width * factor, height * factor)
    new_mask = [cv2.resize(image_mask, dsize, cv2.INTER_NEAREST) for image_mask in mask]

    skeletonized_mask = np.array([skeleton / 255 for skeleton in skeletonize_3d(np.array(new_mask))]).astype('uint8')

    if dilation_kernel is not None:
        skeletonized_mask = [skimage.morphology.binary_dilation(skeleton, dilation_kernel) for skeleton in
                             skeletonized_mask]

    return np.array(skeletonized_mask).astype('uint16') * globals.max_gray_value
