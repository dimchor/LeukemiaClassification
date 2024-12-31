import kagglehub
from skimage import color, filters
from skimage.color.colorconv import rgb2gray
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from matplotlib import pyplot as plt
import csv
from pathlib import Path
import tensorflow as tf
import os
import numpy as np

from utilities import image_loader, label
from cnn import CNN

# Download latest version of dataset
DATASET_PATH = kagglehub.dataset_download("andrewmvd/leukemia-classification")
TRANSFORMS = [
    rgb2gray,
    lambda img: resize(img, (128, 128), anti_aliasing=True),
    img_to_array,
    lambda img: img / 255.0,
]
ALL = 1
HEALTHY = 0
BATCH_SIZE = 64


def main():
    dataset_path = Path(DATASET_PATH)

    # Training set image loader
    il_train = image_loader(
        [
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_0/all/") + os.sep,
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_0/hem/") + os.sep,
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_1/all/") + os.sep,
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_1/hem/") + os.sep,
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_2/all/") + os.sep,
            str(dataset_path / "C-NMC_Leukemia/training_data/fold_2/hem/") + os.sep,
            # str(Path("C:/Users/Dimi/Desktop/training/")) + os.sep
        ],
        TRANSFORMS,
    )
    # Validation set image loader
    il_validation = image_loader(
        [
            str(
                dataset_path
                / "C-NMC_Leukemia/validation_data"
                / "C-NMC_test_prelim_phase_data/"
            )
            + os.sep
            # str(Path("C:/Users/Dimi/Desktop/validation/")) + os.sep
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

    # Generators
    def training_gen():
        images_batch = []
        labels_batch = []
        counter = 0

        for file, image in il_train:
            images_batch.append(image)
            labels_batch.append(ALL if label(file)["label"] == "all" else HEALTHY)
            counter += 1

            if counter == BATCH_SIZE:
                yield np.array(images_batch), np.array(labels_batch)
                images_batch = []
                labels_batch = []
                counter = 0

    def validating_gen():
        images_batch = []
        labels_batch = []
        counter = 0

        for file, image in il_validation:
            images_batch.append(image)
            labels_batch.append(ALL if validation_labels[file] == "1" else HEALTHY)
            counter += 1

            if counter == BATCH_SIZE:
                yield np.array(images_batch), np.array(labels_batch)
                images_batch = []
                labels_batch = []
                counter = 0

    # tf.config.list_physical_devices('GPU')

    training_dataset = tf.data.Dataset.from_generator(
        training_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    validating_dataset = tf.data.Dataset.from_generator(
        validating_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

    model = CNN()

    model.fit(training_dataset)

    loss, accuracy = model.evaluate(validating_dataset)

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
