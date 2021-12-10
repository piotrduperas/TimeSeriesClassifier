import subprocess
import sys

dataset = sys.argv[1]
datasetid = dataset.lower()

result = subprocess.run(
    ['wget', f'https://www.timeseriesclassification.com/Downloads/{dataset}.zip', '-O', f'datasets/{dataset}.zip'],
    stdout=subprocess.PIPE)
print(result.stdout)

subprocess.run(['mkdir', f'datasets/{datasetid}'], stdout=subprocess.PIPE)
result = subprocess.run(['unzip', '-o', f'datasets/{dataset}.zip', '-d', f'datasets/{datasetid}'], stdout=subprocess.PIPE)
print(result.stdout)

subprocess.run(['python3', 'convert.py', datasetid], stdout=subprocess.PIPE)
result = subprocess.run(['python3', 'generate_model.py', datasetid], stdout=subprocess.PIPE)
print(result.stdout)

result = subprocess.run(['python3', 'classify.py', datasetid])
print(result.stdout)
