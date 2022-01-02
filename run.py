import glob
import os
import requests
import shutil
import sys
import zipfile


from io import BytesIO
from os import path
from typing import Tuple

import convert
import generate_model
import classify


def run(dataset: str) -> Tuple[float, float]:
    if path.exists(path.join("datasets", dataset)):
        shutil.rmtree(path.join("datasets", dataset))
    os.mkdir(path.join("datasets", dataset))

    response = requests.get(f"https://www.timeseriesclassification.com/Downloads/{dataset}.zip", stream=True)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    zip_file.extractall(path.join("datasets", dataset))

    contents = glob.glob(path.join("datasets", dataset, '*'))
    if len(contents) == 1 and path.isdir(path.join(contents[0])):
        for filepath in glob.glob(path.join(contents[0], '*')):
            shutil.move(filepath, path.join("datasets", dataset))
        os.rmdir(contents[0])

    convert.run(dataset)
    generate_model.run(dataset)
    return classify.run(dataset)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SyntaxError("Usage: python3 run.py dataset")
    run(sys.argv[1])
