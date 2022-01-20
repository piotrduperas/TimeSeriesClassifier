import math
import numpy
import shutil
import sys

from os import path
from pathlib import Path
from scipy.io import arff
from typing import Tuple


def load_and_save(input_file: str, output_dir: str) -> None:
    data, metadata = arff.loadarff(input_file)
    categories: Tuple[any] = metadata._attributes[list(metadata._attributes.keys())[-1]].values

    file_number = 1
    if isinstance(data[0][0], numpy.floating):
        for inputs in data:
            category = categories.index(inputs[-1].decode("utf-8")) + 1
            values = numpy.array([0 if math.isnan(y) else y for y in numpy.array([x for x in inputs][:-1]).astype("float64")]).transpose()
            values_joined = '\n'.join([str(x) for x in values])

            Path(path.join(output_dir, str(category))).mkdir(parents=True, exist_ok=True)
            with open(path.join(output_dir, str(category), f"{file_number}.csv"), "w+") as output:
                output.writelines(values_joined)

            file_number += 1
    else:
        for inputs in data:
            category = categories.index(inputs[-1].decode("utf-8")) + 1
            values = numpy.vstack([numpy.array([0 if math.isnan(y) else y for y in x]) for x in inputs[0]]).transpose()
            values_joined = [f"{','.join([str(y) for y in x])}\n" for x in values]

            Path(path.join(output_dir, str(category))).mkdir(parents=True, exist_ok=True)
            with open(path.join(output_dir, str(category), f"{file_number}.csv"), "w+") as output:
                output.writelines(values_joined)

            file_number += 1


def run(model_name: str):
    print("Converting...")
    if path.exists(path.join("data", model_name)):
        shutil.rmtree(path.join("data", model_name))

    load_and_save(path.join("datasets", model_name, f"{model_name}_TRAIN.arff"), path.join("data", model_name, "train"))
    load_and_save(path.join("datasets", model_name, f"{model_name}_TEST.arff"), path.join("data", model_name, "test"))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SyntaxError("Usage: python3 convert.py model_name")
    run(sys.argv[1])
