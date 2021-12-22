import ast
import imageio
import keras
import numpy
import sys
from PIL import Image


from os import path
from typing import List


if len(sys.argv) != 2:
    raise SyntaxError("Usage: python3 classify.py model_name")

print("Classifying...")

model = keras.models.load_model(path.join("models", sys.argv[1], "model"))
with open(path.join("models", sys.argv[1], "test_data")) as file:
    image_data = ast.literal_eval(file.read())
    image_paths: List[str] = image_data["test_images"]
    category_count: int = image_data["category_count"]

im_x = []
im_y = []

for image_path in image_paths:
    im = imageio.imread(image_path)
    image_name = path.split(image_path)[-1].split('.')[0]
    image_category = image_name.split('_')[1]
    im_x.append(im[:, :])
    im_y.append(int(image_category) - 1)

im = Image.open(image_paths[0])
IMAGE_WIDTH, IMAGE_HEIGHT = im.size

im_x = numpy.array(im_x)
im_x = im_x.reshape([im_x.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3])

x_test = im_x.astype("float32")
x_test /= 255

y_test = keras.utils.np_utils.to_categorical(im_y, category_count)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
