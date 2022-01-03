import sys

from typing import Tuple

import convert
import generate_model
import classify


def run(dataset: str) -> Tuple[float, float]:
    convert.run(dataset)
    generate_model.run(dataset)
    return classify.run(dataset)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SyntaxError("Usage: python3 run.py dataset")
    run(sys.argv[1])
