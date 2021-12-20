import glob
import imageio
import keras
import math
import numpy
import os
import shutil
import sys

from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import np_utils
from os import path
from pandas import read_csv
from PIL import Image, ImageDraw
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers
from typing import Tuple, List, Union

# constants
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48


# functions
def cap(value: int, min_value: int, max_value: int) -> int:
    return min(max_value, max(min_value, value))


def clean_up(model_name: str):
    for image_path in glob.glob(path.join("pictures", "*.png")):
        os.remove(image_path)

    if path.exists(path.join("models", model_name)):
        shutil.rmtree(path.join("models", model_name))

    os.mkdir(path.join("models", model_name))


def get_files(directories: List[str], category_count: int) -> List[Tuple[str, str, str]]:
    files = []
    for directory in directories:
        for category in range(1, category_count + 1):
            for (dirpath, dirnames, filenames) in os.walk(path.join(directory, str(category))):
                files.extend([(directory, str(category), filename.split('.')[0]) for filename in filenames])
                break
    return files


def add_differentials(data: numpy.ndarray, dimension_count: int, differential_count: int) -> numpy.ndarray:
    result = data.copy()
    filler = numpy.zeros(dimension_count)

    for i in range(differential_count):
        values_to_differentiate: numpy.ndarray = result[:, -dimension_count:]
        differentials = numpy.diff(values_to_differentiate, axis=0)
        differentials = numpy.vstack([filler, differentials])  # add missing values
        result = numpy.hstack([result, differentials])

    return result


def generate_scaler_array(xmin: float, xmax: float, dimension_count: int, differential_count: int) -> numpy.ndarray:
    base = numpy.array([[xmin], [xmax]])

    v = 14
    first_diff_base = numpy.array([[xmin / v], [xmax / v]])
    next_diff_base = numpy.array([[xmin / v / 3], [xmax / v / 3]])

    array = numpy.array(base)
    first_diff_array = numpy.array(first_diff_base)
    next_diff_array = numpy.array(next_diff_base)

    for i in range(dimension_count - 1):
        array = numpy.hstack([array, base])
        first_diff_array = numpy.hstack([first_diff_array, first_diff_base])
        next_diff_array = numpy.hstack([next_diff_array, next_diff_base])

    if differential_count == 0:
        return array

    array = numpy.hstack([array, first_diff_array])

    if differential_count == 1:
        return array

    for i in range(differential_count - 1):
        array = numpy.hstack([array, next_diff_array])

    return array


def group_data(data: numpy.ndarray) -> numpy.ndarray:
    row_count = len(data)
    counter_column = numpy.array(
        [[round(x / row_count * IMAGE_WIDTH) for x in range(row_count)]]
    ).transpose()
    grouped_data = numpy.hstack([data, counter_column])

    result: numpy.ndarray = numpy.empty([0, data.shape[1]])
    for i in range(IMAGE_WIDTH):
        group = numpy.array(grouped_data[grouped_data[:, -1] == i])[:, :-1]
        if len(group) > 0:
            result = numpy.vstack([result, numpy.average(group, axis=0)])

    return result


def draw_dimension(draw: ImageDraw, data: numpy.ndarray, dimension_count: int, dimension_number: int) -> None:
    row_count = len(data)
    start_angle = (dimension_number - 1) / dimension_count * 360
    end_angle = dimension_number / dimension_count * 360
    d = end_angle - start_angle

    for i in range(row_count):
        row = data[i]

        red = cap(round(row[dimension_number - 1] * 255), 0, 255)
        green = cap(round(row[dimension_number - 1 + dimension_count] * 255), 0, 255)
        blue = cap(round(row[dimension_number - 1 + 2 * dimension_count] * 255), 0, 255)

        draw.pieslice((-IMAGE_WIDTH, -IMAGE_WIDTH, 2*IMAGE_WIDTH, 2*IMAGE_HEIGHT), start_angle + (i / row_count * d), start_angle + ((i + 1) / row_count * d), fill=(red, green, blue))


def generate_images(files: List[Tuple[str, str, str]]):
    directory: str
    category: str
    file: str
    for (directory, category, file) in files:
        dataframe = read_csv(f"{path.join(directory, category, file)}.csv", header=None)
        dimension_count = dataframe.shape[1]

        X_raw = dataframe.values
        X = dataframe.values

        X = add_differentials(X, dimension_count, 2)
        X_raw = add_differentials(X_raw, dimension_count, 2)

        for i in range(2, len(X) - 2):
            X[i] = (X_raw[i] * 2 + X_raw[i - 1] * 1 + X_raw[i + 1] * 1) / 4

        scaler = MinMaxScaler(feature_range=(0, 1))
        xmin = X.min()
        xmax = X.max()

        scaler.fit(generate_scaler_array(xmin, xmax, dimension_count, 2))
        scaled_X: numpy.ndarray = scaler.transform(X)
        grouped_X = group_data(scaled_X)

        out = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(out)

        for i in range(1, dimension_count + 1):
            draw_dimension(draw, grouped_X, dimension_count, i)

        out.save(f"pictures/{path.split(directory)[-1]}_{category}_{file}.png", "PNG")


def prepare_data_for_model(image_paths: List[Union[bytes, str]]):
    im_x = []
    im_y = []

    image_paths.sort(reverse=True)

    for image_path in image_paths:
        im = imageio.imread(image_path)
        image_name = path.split(image_path)[-1].split('.')[0]
        image_category = image_name.split('_')[1]
        im_x.append(im[:, :])
        im_y.append(int(image_category) - 1)

    im_x = numpy.array(im_x)
    im_x = im_x.reshape([im_x.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    x75 = len([p for p in image_paths if "train" in p])

    print(x75)

    x_train = im_x[:x75].astype("float32")
    x_test = im_x[x75:].astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.np_utils.to_categorical(im_y[:x75], category_count)
    y_test = keras.utils.np_utils.to_categorical(im_y[x75:], category_count)

    data = dict()
    data["x_train"] = x_train.tolist()
    data["y_train"] = y_train.tolist()
    data["x_test"] = x_test.tolist()
    data["y_test"] = y_test.tolist()
    data["test_images"] = image_paths[x75:]

    return data


def generate_trained_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(2, 2),
                     activation="relu",
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(category_count, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=["accuracy"])

    callback = EarlyStopping(monitor="val_loss", patience=10)

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=64,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[callback])

    return model


if len(sys.argv) != 2:
    raise SyntaxError("Usage: python3 generate_model.py model_name")

clean_up(sys.argv[1])

directories = [path.join("data", sys.argv[1], "test"), path.join("data", sys.argv[1], "train")]
category_count = len(glob.glob(path.join(directories[0], "*")))
files = get_files(directories, category_count)

generate_images(files)

model_data = prepare_data_for_model(glob.glob(path.join("pictures", "*.png")))
model = generate_trained_model(model_data["x_train"], model_data["y_train"], model_data["x_test"], model_data["y_test"])

model.save(path.join("models", sys.argv[1], "model"))
with open(path.join("models", sys.argv[1], "test_data"), "w") as file:
    file.write(str({"category_count": category_count, "test_images": model_data["test_images"]}))
