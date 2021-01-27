from helpers import *

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

            ConstPixelSpacing = []
            i = 0
            # List the instance file paths
            for p in paths:
                img = dcmread(Path(dataset_path).joinpath(p))
                images.append((img, p))
                if i == 0:
                    ConstPixelSpacing = (
                        float(img.PixelSpacing[0]), float(img.PixelSpacing[1]), float(img.SliceThickness))
                i += 1
                if i == 48:
                    break

            print(ConstPixelSpacing)
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
            globals.flood_fill_tolerance = 0.25
            globals.max_gray_value = np.max(globals.median_images)

            # masks, results = evolutive_flood_fill(denoised_images, globals.flood_fill_tolerance, (76, 70),
            #                                       starting_image=16)
            # skeleton = resize_and_skeleton_3d(masks, globals.skeleton_factor, np.ones((2, 2), np.uint8))
            # original_images = copy.deepcopy(globals.images)

            # globals.segmentations_masks.append(
            #     (masks, {'type': 'flood_fill', 'tolerance': globals.flood_fill_tolerance}))
            # globals.segmentations_results.append((results, {'type': 'result'}))

            subplots_slider(
                [['Original', [image.pixel_array for image, _ in images], {'type': 'original'}],
                 ['Median Filter', denoised_images, {'type': 'median_filter'}],
                 ['Mask', [np.zeros(denoised_images[0].shape)] * len(denoised_images),
                  {'type': 'flood_fill', 'flood_fill_tolerance': globals.flood_fill_tolerance,
                   'seed_tolerance': globals.seed_tolerance}],
                 ['Result', [np.zeros(denoised_images[0].shape)] * len(denoised_images), {'type': 'result'}],
                 ['Skeleton', [np.zeros(denoised_images[0].shape)] * len(denoised_images), {'type': 'skeleton'}]],
                click_handler=select_region, zoom=3)
