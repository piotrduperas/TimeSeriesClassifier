## Time series classifier 

Time series classification is performed using convolutional neural networks to classify images generated from time series.

### How to install dependencies?

Assuming that you have `python3` and `pip3` installed, third-party packages can be installed with:

```
pip3 install -r requirements.txt --upgrade --user
```

### How to run on arbitrary data set?

The program is prepared to easily use data sets in a `*.arff` form that data sets at [timeseriesclassification.com](http://timeseriesclassification.com) have.
1. Having two `*.arff` files named `YourDataSet_TRAIN.arff` and `YourDataSet_TEST.arff`, place them in `datasets/YourDataSet/` directory.
2. Run `python convert.py YourDataSet` to convert `*.arff` files into the internal format used in our classifier.
3. Run `python generate_model.py YourDataSet` to generate images, train the model on the training data set, and save the model fo file.
4. Run `python classify.py YourDataSet` to classify images based on test data set.

Alternatively, you can run the program on a data set from [timeseriesclassification.com](http://timeseriesclassification.com) collection with this command:
```
python run.py YouDataSet
```
Change `YourDataSet` to a name of some data set from the website. This script downloads the data set, decompress it, and perform all the steps above.
Please note that this script might only run on Linux systems.

### How to run all data sets from [timeseriesclassification.com](http://timeseriesclassification.com)?

You can run `python run_everything.py` to test the classifier on every data set from the websites. The data sets are downloaded, all steps from above are performed for each data set, and accuracy results are saved to file.
Please note that this script might only run on Linux systems.
