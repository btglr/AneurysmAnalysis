import skimage.morphology
from helpers import *
from PIL import Image
from resizeimage import resizeimage
import cv2

dataset_path = "ImgTP/Ge1CaroG/MR_3DPCA"
globals.dataset_path = dataset_path
ds = load_dataset(dataset_path)

# Iterate through the PATIENT records
for patient in ds.patient_records:
    # Find all the STUDY records for the patient
    studies = [ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"]
    for study in studies:
        # Find all the SERIES records in the study
        all_series = [ii for ii in study.children if ii.DirectoryRecordType == "SERIES"]
        for series in all_series:
            # Find all the IMAGE records in the series
            images = [ii for ii in series.children if ii.DirectoryRecordType == "IMAGE"]

            # Get the absolute file path to each instance
            #   Each IMAGE contains a relative file path to the root directory
            elems = [ii["ReferencedFileID"] for ii in images]
            # Make sure the relative file path is always a list of str
            paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
            paths = [Path(*p) for p in paths]

            images = []
            denoised_images = []

            i = 0
            # List the instance file paths
            for p in paths:
                img = dcmread(Path(dataset_path).joinpath(p))
                images.append((img, p))

                i += 1
                if i == 48:
                    break

            for image, p in images:
                median = apply_simple_denoise(image, kernel_size=3)

                denoised_images.append(median)

                median_dcm = copy.deepcopy(image)
                median_dcm.PixelData = median.tobytes()

                filename = p.name
                study_folder = p.parent.parent
                result_folder = Path(globals.dataset_path).joinpath(study_folder).joinpath('median')
                result_folder.mkdir(parents=True, exist_ok=True)

                filepath = result_folder.joinpath(filename)
                median_dcm.save_as(filepath)

            globals.images = copy.deepcopy(images)
            globals.median_images = denoised_images
            globals.flood_fill_tolerance = 0.31

            mask, result = evolutive_flood_fill(denoised_images, globals.flood_fill_tolerance, (76, 70),
                                                starting_image=16)
            width, height = mask[0].shape
            dsize = (width*10, height*10)

            mask = [cv2.resize(image_mask, dsize) for image_mask in mask]
            skeleton = [skimage.morphology.skeletonize(image_mask) for image_mask in mask]
            subplots_slider(
                [['Original', [image.pixel_array for image, _ in images], {'type': 'original'}],
                 ['Median Filter', denoised_images, {'type': 'median_filter'}],
                 ['Mask', mask, {'type': 'flood_fill', 'tolerance': globals.flood_fill_tolerance}],
                 ['Squelette', skeleton, {'type': 'result'}]],
                click_handler=select_region, zoom=3)