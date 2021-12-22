import subprocess
import json

f = open('datasetTable.json')
 
data = json.load(f)
results = {}

ok = False

 
for i in data:
    dataset = i['Dataset']

    result = subprocess.run(['python3', 'run.py', dataset], stdout=subprocess.PIPE)
    # Read all from result stdout
    try:
        acc = str(result.stdout.decode('utf-8')).split('Test accuracy: ')[1].split('\n')[0]
        results[dataset] = acc
    except:
        results[dataset] = '0'
        print("error")
    with open('results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
