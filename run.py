import subprocess
import sys

dataset = sys.argv[1]
datasetid = dataset.lower()

result = subprocess.run(['wget', 'https://www.timeseriesclassification.com/Downloads/'+dataset+'.zip', '-O', 'datasets/'+dataset+'.zip'], stdout=subprocess.PIPE)
print(result.stdout)

result = subprocess.run(['mkdir', 'datasets/'+datasetid], stdout=subprocess.PIPE)
result = subprocess.run(['unzip', '-o', 'datasets/'+dataset+'.zip', '-d', 'datasets/'+datasetid], stdout=subprocess.PIPE)
print(result.stdout)
result = subprocess.run(['python3', 'convert.py', 'datasets/'+datasetid+'/'+dataset+'_TRAIN.arff', 'datasets/'+datasetid+'/'+dataset+'_TEST.arff', 'datasets/'+datasetid+'/converted'], stdout=subprocess.PIPE)
result = subprocess.run(['python3', 'generate_model.py', 'datasets/'+datasetid+'/converted'], stdout=subprocess.PIPE)
result = subprocess.run(['python3', 'generate_model.py', 'datasets/'+datasetid+'/converted'], stdout=subprocess.PIPE)
print(result.stdout)
