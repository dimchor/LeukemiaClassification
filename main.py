import kagglehub
from scipy.ndimage import value_indices
from skimage import color, filters, exposure
from skimage.color.colorconv import rgb2gray
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from matplotlib import pyplot as plt
import csv
from pathlib import Path
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

from utilities import image_loader, label, rows
from cnn import CNN

# Download latest version of dataset
DATASET_PATH = kagglehub.dataset_download("andrewmvd/leukemia-classification")
TRANSFORMS = [
    rgb2gray,
    lambda img: resize(img, (128, 128), anti_aliasing=True),
    exposure.equalize_hist,
    img_to_array,
    lambda img: img / 255.0,
]
ALL = 1
HEALTHY = 0
BATCH_SIZE = 40
EPOCHS = 20


def load_validation_labels():
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
    return validation_labels


def train_model(validation_labels):
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
            il_validation = image_loader(
                DATASET_PATH, "validation.txt", TRANSFORMS
            )
            for file, image in il_validation:
                images.append(image)
                labels.append(
                    ALL if validation_labels[file] == "1" else HEALTHY
                )
                counter += 1
                if counter < BATCH_SIZE:
                    continue
                yield np.array(images), np.array(labels)
                images = []
                labels = []
                counter = 0

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )

    checkpoint = ModelCheckpoint(
        "./checkpoints/checkpoint_" + str(int(time.time())) + ".keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    model = CNN((128, 128, 1))

    history = model.fit(
        training_gen(),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        steps_per_epoch=rows("training.txt") // BATCH_SIZE,
        validation_data=validating_gen(),
        validation_steps=rows("validation.txt") // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
    )

    model.save_weights_history()


def evaluate_model(validation_labels):
    def testing_gen():
        images = []
        labels = []
        counter = 0

        while True:
            # Testing set image loader
            il_validation = image_loader(
                DATASET_PATH, "testing.txt", TRANSFORMS
            )
            for file, image in il_validation:
                images.append(image)
                labels.append(
                    ALL if validation_labels[file] == "1" else HEALTHY
                )
                counter += 1
                if counter < BATCH_SIZE:
                    continue
                yield np.array(images), np.array(labels)
                images = []
                labels = []
                counter = 0

    model = CNN((128, 128, 1))

    model.load_weights("./sample_weights/model_1735827065.weights.h5")

    loss, accuracy = model.evaluate(
        testing_gen(), rows("testing.txt"), BATCH_SIZE
    )

    all_cases = sum(
        [1 if key == "1" else 0 for key in validation_labels.values()]
    ) / len(validation_labels)
    print(f"ALL cases: {all_cases}")
    print(f"HEALTY individuals: {1 - all_cases}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    validation_labels = load_validation_labels()
    evaluate_model(validation_labels)
