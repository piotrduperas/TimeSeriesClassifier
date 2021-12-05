import glob
import imageio
import keras
import numpy
import os
import random
import shutil
import sys

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from os import path
from pandas import read_csv
from PIL import Image, ImageDraw
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers
from typing import Tuple, List, Union

# constants
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


# functions
def clean_up(model_name: str):
    for image_path in glob.glob(path.join("pictures", "*.png")):
        os.remove(image_path)

    if path.exists(path.join("models", model_name)):
        shutil.rmtree(path.join("models", model_name))

    os.mkdir(path.join("models", model_name))
    os


def get_files(directories: List[str], category_count: int):
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


def generate_images(files: List[Tuple[str, str, any]]):
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
        scaled_X = scaler.transform(X)

        out = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))

        # get a drawing context
        d = ImageDraw.Draw(out)

        mid = IMAGE_HEIGHT // 2 - 1

        for ii in range(0, len(scaled_X)):
            i = len(scaled_X) - 1 - ii
            row = scaled_X[i]

            intensity = 255 - i * 8 // 9
            intensity = intensity // 3
            x_row = 2 * dimension_count
            y_row = 2 * dimension_count

            x, y = (row[x_row] * IMAGE_HEIGHT % IMAGE_HEIGHT), mid

            # red
            current_color = out.getpixel((x, mid))
            d.point((x, mid), fill=(intensity, current_color[1], current_color[2]))

            if dimension_count == 1:
                continue

            # green
            current_color = out.getpixel((((row[x_row + 1] * IMAGE_HEIGHT) % IMAGE_HEIGHT), mid))
            d.point((row[x_row + 1] * IMAGE_HEIGHT, mid),
                    fill=(current_color[0], intensity, current_color[2]))

            if dimension_count == 2:
                continue

            # blue
            current_color = out.getpixel(((row[x_row + 2] * IMAGE_HEIGHT % IMAGE_HEIGHT), mid))
            d.point((row[x_row + 2] * IMAGE_HEIGHT, mid),
                    fill=(current_color[0], current_color[1], intensity))

        for ii in range(0, len(scaled_X)):
            i = len(scaled_X) - 1 - ii
            row = scaled_X[i]

            intensity = 255 - i * 8 // 9
            x_row = dimension_count
            y_row = dimension_count

            x, y = (row[x_row] * IMAGE_HEIGHT % IMAGE_HEIGHT), (row[y_row] * IMAGE_HEIGHT) % IMAGE_HEIGHT

            # red
            current_color = out.getpixel((x, IMAGE_HEIGHT - 1 - y))
            d.point((x, IMAGE_HEIGHT - 1 - y), fill=(intensity, current_color[1], current_color[2]))

            if dimension_count == 1:
                continue

            # green
            current_color = out.getpixel((((row[x_row + 1] * IMAGE_HEIGHT) % IMAGE_HEIGHT),
                                          IMAGE_HEIGHT - 1 - ((row[y_row + 1] * IMAGE_HEIGHT)) % IMAGE_HEIGHT))
            d.point((row[x_row + 1] * IMAGE_HEIGHT, IMAGE_HEIGHT - 1 - row[y_row + 1] * IMAGE_HEIGHT),
                    fill=(current_color[0], intensity, current_color[2]))

            if dimension_count == 2:
                continue

            # blue
            current_color = out.getpixel(((row[x_row + 2] * IMAGE_HEIGHT % IMAGE_HEIGHT),
                                          IMAGE_HEIGHT - 1 - (row[y_row + 2] * IMAGE_HEIGHT) % IMAGE_HEIGHT))
            d.point((row[x_row + 2] * IMAGE_HEIGHT, IMAGE_HEIGHT - 1 - row[(y_row + 2)] * IMAGE_HEIGHT),
                    fill=(current_color[0], current_color[1], intensity))

        for ii in range(0, len(scaled_X)):
            i = len(scaled_X) - 1 - ii
            row = scaled_X[i]

            intensity = 255 - i * 8 // 9
            x_row = 0
            y_row = 0

            x, y = (row[x_row] * IMAGE_HEIGHT % IMAGE_HEIGHT), (row[y_row] * IMAGE_HEIGHT) % IMAGE_HEIGHT

            # red
            current_color = out.getpixel((x, y))
            d.point((x, y), fill=(intensity, current_color[1], current_color[2]))

            if dimension_count == 1:
                continue

            # green
            current_color = out.getpixel(
                (((row[x_row + 1] * IMAGE_HEIGHT) % IMAGE_HEIGHT), ((row[y_row + 1] * IMAGE_HEIGHT)) % IMAGE_HEIGHT))
            d.point((row[x_row + 1] * IMAGE_HEIGHT, row[y_row + 1] * IMAGE_HEIGHT),
                    fill=(current_color[0], intensity, current_color[2]))

            if dimension_count == 2:
                continue

            # blue
            current_color = out.getpixel(
                ((row[x_row + 2] * IMAGE_HEIGHT % IMAGE_HEIGHT), (row[y_row + 2] * IMAGE_HEIGHT) % IMAGE_HEIGHT))
            d.point((row[x_row + 2] * IMAGE_HEIGHT, row[(y_row + 2)] * IMAGE_HEIGHT),
                    fill=(current_color[0], current_color[1], intensity))

        out.save(f"pictures/{category}_{file}.png", "PNG")


def prepare_data_for_model(image_paths: List[Union[bytes, str]]):
    im_x = []
    im_y = []

    random.shuffle(image_paths)

    for image_path in image_paths:
        im = imageio.imread(image_path)
        image_name = path.split(image_path)[-1].split('.')[0]
        image_category = image_name.split('_')[0]
        im_x.append(im[:, :])
        im_y.append(int(image_category) - 1)

    im_x = numpy.array(im_x)
    im_x = im_x.reshape([im_x.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    x75 = int(0.75 * len(im_x))

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

    return data


def generate_trained_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation="relu",
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(category_count, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=64,
              verbose=1,
              validation_data=(x_test, y_test))

    return model


if len(sys.argv) != 3:
    raise SyntaxError("Usage: python3 generate_model.py data_directory model_name")

clean_up(sys.argv[2])

directories = [f"{sys.argv[1]}/test", f"{sys.argv[1]}/train"]
category_count = len(glob.glob(path.join(directories[0], "*")))
files = get_files(directories, category_count)

generate_images(files)

model_data = prepare_data_for_model(glob.glob(path.join("pictures", "*.png")))
model = generate_trained_model(model_data["x_train"], model_data["y_train"], model_data["x_test"], model_data["y_test"])

model.save(path.join("models", sys.argv[2], "model"))
with open(path.join("models", sys.argv[2], "model_data"), "w") as file:
    file.write(str(model_data))
