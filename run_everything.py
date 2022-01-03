import json
import download_and_run


def run():
    with open("datasetTable.json") as f:
        data = json.load(f)
        results = {}

        for i in data:
            dataset = i['Dataset']

            try:
                loss, accuracy = download_and_run.run(dataset)
                results[dataset] = accuracy
            except:
                results[dataset] = "0"
                print("error")

            with open('results.json', 'w') as outfile:
                json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    run()
