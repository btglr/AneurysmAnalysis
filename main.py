import os

from pydicom import dcmread

from helpers import load_dataset, apply_denoise
from pathlib import Path
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

            images = []
            denoised_images = []

            # List the instance file paths
            for p in paths:
                # print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

                img = dcmread(Path(dataset_path).joinpath(p))
                images.append(img)

                # plt.imshow(img.pixel_array, cmap=plt.cm.bone)
                # plt.show()

            # Iterate over the stored images
            # Denoise them and store them in a new list
            for image in images:
                new_img = apply_denoise(images[0], kernel_size=3)
                denoised_images.append(new_img)

                # plt.imshow(new_img, cmap=plt.cm.bone)
                # plt.show()
