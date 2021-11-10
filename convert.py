import sys

from pathlib import Path

if len(sys.argv) != 4:
    raise SyntaxError("Usage: python3 convert.py train_file.arff test_file.arff output_dir_name")

with open(sys.argv[1], 'r') as file:
    data = False
    file_number = 1
    for line in file:
        if data:
            values = line.split(',')
            category = int(values[-1])
            values = values[:-1]

            Path(f"{sys.argv[3]}/train/{category}").mkdir(parents=True, exist_ok=True)
            with open(f"{sys.argv[3]}/train/{category}/{file_number}.csv", "w+") as output:
                output.writelines(values)

            file_number += 1
        elif line == "@data\n":
            data = True

with open(sys.argv[2], 'r') as file:
    data = False
    file_number = 1
    for line in file:
        if data:
            values = line.split(',')
            category = int(values[-1])
            values = values[:-1]

            Path(f"{sys.argv[3]}/test/{category}").mkdir(parents=True, exist_ok=True)
            with open(f"{sys.argv[3]}/test/{category}/{file_number}.csv", "w+") as output:
                output.writelines(values)

            file_number += 1
        elif line == "@data\n":
            data = True

