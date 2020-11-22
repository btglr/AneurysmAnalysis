import copy
import os

from helpers import *
import matplotlib.pyplot as plt

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

            # List the instance file paths
            for p in paths:
                print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

                img = dcmread(Path(dataset_path).joinpath(p))

                # plt.xlabel("Original")
                # plt.imshow(img.pixel_array, cmap=plt.cm.bone)
                # plt.show()

                median = apply_simple_denoise(img, kernel_size=3)
                median_dcm = copy.deepcopy(img)
                median_dcm.PixelData = median.tobytes()

                # plt.xlabel("Median")
                # plt.imshow(median_dcm.pixel_array, cmap=plt.cm.bone)
                # plt.show()

                nlm = apply_non_local_means(img, kernel=5, window_search=7)
                nlm_dcm = copy.deepcopy(img)
                nlm_dcm.PixelData = nlm.tobytes()

                # plt.xlabel("Non local means")
                # plt.imshow(nlm_dcm.pixel_array, cmap=plt.cm.bone)
                # plt.show()

                bilateral = apply_bilateral_filtering(img, 4, 35, 35)
                bilateral_dcm = copy.deepcopy(img)
                bilateral_dcm.PixelData = bilateral.tobytes()

                # plt.xlabel("Bilateral")
                # plt.imshow(bilateral_dcm.pixel_array, cmap=plt.cm.bone)
                # plt.show()

                median_path = Path(dataset_path).joinpath("{}_median".format(p))
                nlm_path = Path(dataset_path).joinpath("{}_nlm".format(p))
                bilateral_path = Path(dataset_path).joinpath("{}_bilateral".format(p))

                median_dcm.save_as(median_path)
                nlm_dcm.save_as(nlm_path)
                bilateral_dcm.save_as(bilateral_path)
