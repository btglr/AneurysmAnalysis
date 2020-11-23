import copy

from helpers import *

dataset_path = "ImgTP/Ge1CaroG/MR_3DPCA"
ds = load_dataset(dataset_path)

# Iterate through the PATIENT records
for patient in ds.patient_records:
    # print(
    #     f"PATIENT: PatientID={patient.PatientID}, "
    #     f"PatientName={patient.PatientName}"
    # )

    # Find all the STUDY records for the patient
    studies = [
        ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
    ]
    for study in studies:
        descr = study.StudyDescription or "(no value available)"
        # print(
        #     f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
        #     f"StudyDate={study.StudyDate}, StudyDescription={descr}"
        # )

        # Find all the SERIES records in the study
        all_series = [
            ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
        ]
        for series in all_series:
            # Find all the IMAGE records in the series
            images = [
                ii for ii in series.children
                if ii.DirectoryRecordType == "IMAGE"
            ]
            plural = ('', 's')[len(images) > 1]

            descr = getattr(
                series, "SeriesDescription", "(no value available)"
            )
            # print(
            #     f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
            #     f"Modality={series.Modality}, SeriesDescription={descr} - "
            #     f"{len(images)} SOP Instance{plural}"
            # )

            # Get the absolute file path to each instance
            #   Each IMAGE contains a relative file path to the root directory
            elems = [ii["ReferencedFileID"] for ii in images]
            # Make sure the relative file path is always a list of str
            paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
            paths = [Path(*p) for p in paths]

            images = []
            denoised_images = []
            threshold_images = []
            random_walker_images = []
            fills = {}

            i = 0
            # List the instance file paths
            for p in paths:
                # print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

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

                path = Path(dataset_path).joinpath("{}_median".format(p))
                median_dcm.save_as(path)

            for median_image in denoised_images:
                thresh = apply_threshold(median_image)
                random_walker = apply_random_walker(median_image)

                for tol in np.linspace(0.2, 0.4, 12):
                    str_tol = "{:.2f}".format(tol)

                    if str_tol not in fills:
                        fills[str_tol] = []

                    fill = apply_flood_fill(median_image, (76, 69), tol)
                    fills[str_tol].append(fill)

                threshold_images.append(thresh)
                random_walker_images.append(random_walker)

            all_images = []

            all_images.append(('Original', [image.pixel_array for image, _ in images], {'type': 'original'}))
            all_images.append(('Median Filter', denoised_images, {'type': 'median_filter'}))
            all_images.append(('Threshold', threshold_images, {'type': 'threshold'}))
            all_images.append(('Random Walker', random_walker_images, {'type': 'random_walker'}))

            globals.median_images = denoised_images

            for tol in fills:
                all_images.append(
                    ('Flood Fill Tol {}'.format(tol), fills[tol], {'type': 'flood_fill', 'tolerance': float(tol)}))

            subplots_slider(all_images, click_handler=select_region, zoom=1)

            # Select image 16
            image_number = 16
            # Select image set with tolerance 0.31
            selected_tol = fills['0.31']
            new_tol = selected_tol

            # Select the image number from each set
            selected_gray = denoised_images[image_number]
            selected_fill = selected_tol[image_number]

            # Multiply the gray image with the mask
            anevrism = selected_gray * selected_fill

            # Get the average value of the result
            average_gray = int(np.nanmean(nan_if(anevrism, 0)))

            print("Average gray: {}".format(average_gray))

            for image_number in range(17, 48):
                # Apply the mask to the next image
                selected_gray = denoised_images[image_number]
                anevrism = selected_gray * selected_fill

                # Find a new starting point
                close_values = np.where(np.isclose(anevrism, average_gray, atol=20))

                # Combine the two 1D arrays so we get an array of (x, y) coordinates
                combined = np.column_stack(close_values)

                # Take a random starting point
                index = np.random.randint(0, combined.shape[0])

                new_coordinates = (combined[index][0], combined[index][1])
                print("New starting coordinates: ({}, {})".format(combined[index][0], combined[index][1]))

                new_tol[image_number] = apply_flood_fill(selected_gray, new_coordinates, 0.31)
                selected_fill = new_tol[image_number]
                anevrism = selected_gray * selected_fill
                average_gray = int(np.nanmean(nan_if(anevrism, 0)))

            plot_slider(new_tol, "Dynamic")
