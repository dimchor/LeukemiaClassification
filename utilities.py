from typing import Callable
from skimage import io
import os
import re

def image_loader(directories: list, transforms: list[Callable]):
    for directory in directories:
        for file in os.listdir(directory):
            im = io.imread(directory + file)
            for transform in transforms:
                im = transform(im)
            yield file, im


def label(filename):
    result = re.search(r"UID_H?(\d*)_(\d*)_(\d*)_(all|hem)", filename)
    return {
        "id": int(result.group(1)),
        "number": int(result.group(2)),
        "count": int(result.group(3)),
        "label": result.group(4),
    }
