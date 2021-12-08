import ast
import keras
import sys

from os import path

if len(sys.argv) != 2:
    raise SyntaxError("Usage: python3 classify.py model_name")

model = keras.models.load_model(path.join("models", sys.argv[1], "model"))
with open(path.join("models", sys.argv[1], "model_data")) as file:
    model_data = ast.literal_eval(file.read())

score = model.evaluate(model_data["x_test"], model_data["y_test"], verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
