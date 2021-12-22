import csv
import subprocess
import json

# with open("comparison.csv") as fp:
#     reader = csv.reader(fp, delimiter=",", quotechar='"')
#     # next(reader, None)  # skip the headers
#     data_read = [row for row in reader]

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





# for i in range(1, 6):
#     for d in data_read[1:]:
#         result = subprocess.run(['python3', 'run.py', d[0]], stdout=subprocess.PIPE)
#         # Read all from result stdout
#         try:
#             acc = str(result.stdout.decode('utf-8')).split('Test accuracy: ')[1].split('\\n')[0]
#             d.append(acc)
#             print(d)
#         except:
#             print("error")

# with open("results.csv", "wt") as fp:
#     writer = csv.writer(fp, delimiter=",")
#     writer.writerow(data_read[0])  # write header
#     writer.writerows(data_read[1:])
