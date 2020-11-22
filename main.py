import os

from helpers import *

dataset_path = "ImgTP/Ge1CaroG/MR_3DPCA"
ds = load_dataset(dataset_path)

# Iterate through the PATIENT records
for patient in ds.patient_records:
    print(
        f"PATIENT: PatientID={patient.PatientID}, "
        f"PatientName={patient.PatientName}"
    )

    # Find all the STUDY records for the patient
    studies = [
        ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
    ]
    for study in studies:
        descr = study.StudyDescription or "(no value available)"
        print(
            f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
            f"StudyDate={study.StudyDate}, StudyDescription={descr}"
        )

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
            print(
                f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                f"Modality={series.Modality}, SeriesDescription={descr} - "
                f"{len(images)} SOP Instance{plural}"
            )

            # Get the absolute file path to each instance
            #   Each IMAGE contains a relative file path to the root directory
            elems = [ii["ReferencedFileID"] for ii in images]
            # Make sure the relative file path is always a list of str
            paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
            paths = [Path(*p) for p in paths]

            images = []
            denoised_images = []
            non_local__means_images = []
            bilateral_images = []

            i = 0
            # List the instance file paths
            for p in paths:
                print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

                img = dcmread(Path(dataset_path).joinpath(p))
                images.append((img, p))

                i += 1
                if i == 24:
                    break

            for image, p in images:
                median = apply_simple_denoise(image, kernel_size=3)

                denoised_images.append(median)

                # median_dcm = copy.deepcopy(image)
                # median_dcm.PixelData = median.tobytes()
                #
                # path = Path(dataset_path).joinpath("{}_median".format(p))
                # median_dcm.save_as(path)

            for image, p in images:
                nlm = apply_non_local_means(image)
                non_local__means_images.append(nlm)

                # nlm_dcm = copy.deepcopy(image)
                # nlm_dcm.PixelData = nlm.tobytes()
                #
                # path = Path(dataset_path).joinpath("{}_nlm".format(p))
                # nlm_dcm.save_as(path)

            for image, p in images:
                bilateral = apply_bilateral_filtering(image, 4, 35, 35)
                bilateral_images.append(bilateral)

                # bilateral_dcm = copy.deepcopy(image)
                # bilateral_dcm.PixelData = bilateral.tobytes()
                #
                # path = Path(dataset_path).joinpath("{}_bilateral".format(p))
                # bilateral_dcm.save_as(path)

            subplots_slider(
                [("Original", [image.pixel_array for image, _ in images]), ("Median Filter", denoised_images),
                 ("Non Local Means", non_local__means_images),
                 ("Bilateral Filter", bilateral_images)])
