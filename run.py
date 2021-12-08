import subprocess
import sys

dataset = sys.argv[1]
datasetid = dataset.lower()

result = subprocess.run(
    ['wget', f'https://www.timeseriesclassification.com/Downloads/{dataset}.zip', '-O', f'datasets/{dataset}.zip'],
    stdout=subprocess.PIPE)
print(result.stdout)

subprocess.run(['mkdir', f'datasets/{datasetid}'], stdout=subprocess.PIPE)
result = subprocess.run(['unzip', '-o', 'datasets/{dataset}.zip', '-d', 'datasets/{datasetid}'], stdout=subprocess.PIPE)
print(result.stdout)

subprocess.run(
    ['python3', 'convert.py', f'datasets/{datasetid}/{dataset}_TRAIN.arff', f'datasets/{datasetid}/{dataset}_TEST.arff',
     f'datasets/{datasetid}/converted'], stdout=subprocess.PIPE)
result = subprocess.run(['python3', 'generate_model.py', f'datasets/{datasetid}/converted', datasetid],
                        stdout=subprocess.PIPE)
print(result.stdout)

result = subprocess.run(['python3', 'classify.py', datasetid])
