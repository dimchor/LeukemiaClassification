from typing import Callable
from skimage import io
import os
import re
from pathlib import Path

def image_loader(path: str, filename: str, transforms: list[Callable]):
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            im = io.imread(Path(path + line))
            for transform in transforms:
                im = transform(im)
            yield Path(line).name, im


def label(filename):
    result = re.search(r"UID_H?h?(\d+)_(\d+)_(\d+)_(all|hem)", filename)
    return {
        "id": int(result.group(1)),
        "number": int(result.group(2)),
        "count": int(result.group(3)),
        "label": result.group(4),
    }
