import csv
import subprocess

with open("comparison.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]

for i in range(1, 6):
    for d in data_read[1:]:
        result = subprocess.run(['python3', 'run.py', d[0]], stdout=subprocess.PIPE)
        # Read all from result stdout
        try:
            acc = str(result.stdout.decode('utf-8')).split('Test accuracy: ')[1].split('\\n')[0]
            d.append(acc)
            print(d)
        except:
            print("error")

with open("results.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(data_read[0])  # write header
    writer.writerows(data_read[1:])
