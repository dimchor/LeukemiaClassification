import kagglehub
from skimage import color, filters
from skimage.color.colorconv import rgb2gray
from matplotlib import pyplot as plt
import csv
from pathlib import Path

from utilities import image_loader, label

# Download latest version of dataset
DATASET_PATH = kagglehub.dataset_download("andrewmvd/leukemia-classification")
TRANSFORMS = [rgb2gray]
ALL = 1
HEALTHY = 0


def main():
    dataset_path = Path(DATASET_PATH)

    il_train = image_loader(
        [
            dataset_path / "C-NMC_Leukemia/training_data/fold_0/all/",
            dataset_path / "C-NMC_Leukemia/training_data/fold_0/hem/",
            dataset_path / "C-NMC_Leukemia/training_data/fold_1/all/",
            dataset_path / "C-NMC_Leukemia/training_data/fold_1/hem/",
            dataset_path / "C-NMC_Leukemia/training_data/fold_2/all/",
            dataset_path / "C-NMC_Leukemia/training_data/fold_2/hem/",
        ],
        TRANSFORMS,
    )
    il_validation = image_loader(
        [
            dataset_path
            / "C-NMC_Leukemia/validation_data"
            / "C-NMC_test_prelim_phase_data/"
        ],
        TRANSFORMS,
    )

    validation_labels = {}
    with open(
        dataset_path
        / "C-NMC_Leukemia/validation_data"
        / "C-NMC_test_prelim_phase_data_labels.csv",
        newline="",
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            validation_labels[row["new_names"]] = row["labels"]


if __name__ == "__main__":
    main()
