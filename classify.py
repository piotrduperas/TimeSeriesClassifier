import glob
import imageio
import keras
import numpy
import os
import random

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.utils import np_utils
from os import path
from pandas import read_csv
from PIL import Image, ImageDraw
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers

def add_differentials(data: numpy.ndarray, differential_count: int) -> numpy.ndarray:
    result = data.copy()
    filler = numpy.zeros(3)

    for i in range(differential_count):
        values_to_differentiate: numpy.ndarray = result[:, -3:]
        differentials = numpy.diff(values_to_differentiate, axis=0)
        differentials = numpy.vstack([filler, differentials])     # add missing values
        result = numpy.hstack([result, differentials])

    return result


# constants
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CATEGORY_COUNT = 8

directories = ["data/test", "data/train"]

files = []

for directory in directories:
    for category in range(1, CATEGORY_COUNT + 1):
        for (dirpath, dirnames, filenames) in os.walk(path.join(directory, str(category))):
            files.extend([(directory, str(category), filename.split('.')[0]) for filename in filenames])
            break

for (directory, category, file) in files:
    dataframe = read_csv(f"{path.join(directory, category, file)}.csv", header=None)
    X_raw = dataframe.values
    X = dataframe.values

    for i in range(2, len(X) - 2):
      X[i] = (X[i] * 4 + X[i-1] * 2 + X[i+1] * 2 + X[i-2] + X[i+2]) / 10;

    X = add_differentials(X, 2)

    v = 12

    scaler = MinMaxScaler(feature_range=(0, 1))
    xmin = X.min()
    xmax = X.max()
    scaler.fit(numpy.array([
        # x     y     z      dx         dy         dz         ddx        ddy        ddz
        [xmin, xmin, xmin, xmin / v, xmin / v, xmin / v, xmin / v, xmin / v, xmin / v],
        [xmax, xmax, xmax, xmax / v, xmax / v, xmax / v, xmax / v, xmax / v, xmax / v]
    ]))
    scaled_X = scaler.transform(X)

    out = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))

    # get a drawing context
    d = ImageDraw.Draw(out)

    for i in range(0, len(scaled_X)):
        row = scaled_X[i]

        intensity = 255 - i * 5 // 6
        #if intensity < 5:
        #  intensity = 5
        x_row = 0
        y_row = 3

        # x and ddx
        x, y = (row[x_row] * IMAGE_HEIGHT % IMAGE_HEIGHT), (row[y_row] * IMAGE_HEIGHT) % IMAGE_HEIGHT

        # red
        current_color = out.getpixel((x, y))
        d.point((x, y), fill=(intensity, current_color[1], current_color[2]))

        # green
        current_color = out.getpixel((((row[x_row + 1] * IMAGE_HEIGHT) % IMAGE_HEIGHT), ((row[y_row + 1] * IMAGE_HEIGHT)) % IMAGE_HEIGHT))
        d.point((row[x_row + 1] * IMAGE_HEIGHT, row[y_row + 1] * IMAGE_HEIGHT),
                fill=(current_color[0], intensity, current_color[2]))

        # blue
        current_color = out.getpixel(((row[x_row + 2] * IMAGE_HEIGHT % IMAGE_HEIGHT), (row[y_row + 2] * IMAGE_HEIGHT) % IMAGE_HEIGHT))
        d.point((row[x_row + 2] * IMAGE_HEIGHT, row[(y_row + 2)] * IMAGE_HEIGHT),
                fill=(current_color[0], current_color[1], intensity))

    out.save(f"pictures/{category}_{file}.png", "PNG")


im_x = []
im_y = []


image_paths = glob.glob("pictures/*.png")
random.shuffle(image_paths)

for image_path in image_paths:
    im = imageio.imread(image_path)
    image_name = path.split(image_path)[-1].split('.')[0]
    image_category = image_name.split('_')[0]
    im_x.append(im[:, :])
    im_y.append(int(image_category) - 1)


im_x = numpy.array(im_x)
im_x = im_x.reshape([im_x.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3])


x_train = im_x[:2500].astype("float32")
x_test = im_x[2500:].astype("float32")
x_train /= 255
x_test /= 255

num_classes = 8

y_train = keras.utils.np_utils.to_categorical(im_y[:2500], num_classes)
y_test = keras.utils.np_utils.to_categorical(im_y[2500:], num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
model.add(Conv2D(128, (4, 4), activation="relu"))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=24,
          verbose=1,
          validation_data=(x_test, y_test))