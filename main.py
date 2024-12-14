import kagglehub
from typing import Callable
from skimage import io, color, filters
from skimage.color.colorconv import rgb2gray
from matplotlib import pyplot as plt
import os


# Download latest version of dataset
DATASET_PATH = kagglehub.dataset_download("andrewmvd/leukemia-classification")
TRANSFORMS = [
    rgb2gray
]

def image_loader(directories: list[str], transforms: list[Callable]):
    for directory in directories:
        for file in os.listdir(directory):
            im = io.imread(directory + file)
            for transform in transforms:
                im = transform(im)
            yield im

def main():
    for image in image_loader(['C:\\Users\\Dimi\\Desktop\\images\\'], TRANSFORMS):
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main()
