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
# DATASET_PATH = kagglehub.dataset_download("andrewmvd/leukemia-classification")
DATASET_PATH = "C:\\Users\\Dimi\\.cache\\kagglehub\\datasets\\andrewmvd\\leukemia-classification\\versions\\2"
TRANSFORMS = [
    rgb2gray,
    lambda img: resize(img, (128, 128), anti_aliasing=True),
    img_to_array,
    lambda img: img / 255.0,
]
ALL = 1
HEALTHY = 0
BATCH_SIZE = 80
EPOCHS = 10


def main():
    validation_labels = {}
    with open(
        Path(DATASET_PATH)
        / "C-NMC_Leukemia/validation_data"
        / "C-NMC_test_prelim_phase_data_labels.csv",
        newline="",
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            validation_labels[row["new_names"]] = row["labels"]

    # Generators
    def training_gen():
        images = []
        labels = []
        counter = 0

        while True:
            # Training set image loader
            il_train = image_loader(DATASET_PATH, "training.txt", TRANSFORMS)
            for file, image in il_train:
                images.append(image)
                labels.append(ALL if label(file)["label"] == "all" else HEALTHY)
                counter += 1
                if counter < BATCH_SIZE:
                    continue
                yield np.array(images), np.array(labels)
                images = []
                labels = []
                counter = 0

    def validating_gen():
        images = []
        labels = []
        counter = 0

        while True:
            # Validation set image loader
            il_validation = image_loader(DATASET_PATH, "validation.txt", TRANSFORMS)
            for file, image in il_validation:
                images.append(image)
                labels.append(ALL if validation_labels[file] == "1" else HEALTHY)
                counter += 1
                if counter < BATCH_SIZE:
                    continue
                yield np.array(images), np.array(labels)
                images = []
                labels = []
                counter = 0


    training_dataset = tf.data.Dataset.from_generator(
        training_gen,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
        ),
    )

    validating_dataset = tf.data.Dataset.from_generator(
        validating_gen,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
        ),
    )


    model = CNN(BATCH_SIZE, EPOCHS)

    # model.fit(training_dataset)

    model.load("checkpoint_1735663369.weights.h5")

    loss, accuracy = model.evaluate(validating_gen())

    # model.save()

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
