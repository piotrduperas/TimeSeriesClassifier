import csv
import json

with open("comparison.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]

f = open('results.json')
 
results = json.load(f)

data_read[0].append("Our results")

for i in data_read[1:]:
    dataset = i[0]
    if dataset in results and results[dataset] != "0":
        i.append(results[dataset])

with open("results_all.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(data_read[0])  # write header
    writer.writerows(data_read[1:])
