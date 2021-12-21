import subprocess
import sys

dataset = sys.argv[1]
datasetid = dataset.lower()

result = subprocess.run(
    ['wget', f'https://www.timeseriesclassification.com/Downloads/{dataset}.zip', '-O', f'datasets/{dataset}.zip'],
    stdout=subprocess.PIPE)
print(result.stdout)

subprocess.run(['mkdir', f'datasets/{dataset}'], stdout=subprocess.PIPE)
result = subprocess.run(['unzip', '-o', f'datasets/{dataset}.zip', '-d', f'datasets/{dataset}'], stdout=subprocess.PIPE)
print(result.stdout)
subprocess.run(['rm', f'datasets/{dataset}.zip'], stdout=subprocess.PIPE)

subprocess.run(['python3', 'convert.py', dataset], stdout=subprocess.PIPE)
result = subprocess.run(['python3', 'generate_model.py', dataset], stdout=subprocess.PIPE)
print(result.stdout)

result = subprocess.run(['python3', 'classify.py', dataset])
print(result.stdout)
