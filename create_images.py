import glob
import imageio
import keras
import numpy
import os

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from os import path
from pandas import read_csv
from PIL import Image, ImageDraw
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers

# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D


s = 64
w = s
h = s

directories = ['data/test/', 'data/train/']

f = []

for directory in directories:
    for category in range(1, 9):
        for (dirpath, dirnames, filenames) in os.walk(path.join(directory, str(category))):
            f.extend([(directory, str(category), s.split('.')[0]) for s in filenames])
            break

for (directory, category, file) in f:
# uncomment lines below to prevent files generation
#    pass
#for i in range(1,1):
    dataframe = read_csv(directory + category + '/' + file + '.csv', header=None)
    array = dataframe.values
    # separate array into input and output components
    X = array[:]
    diff = numpy.diff(X, axis=0)
    aaa = numpy.vstack((diff, numpy.array([[0, 0, 0]])))

    X = numpy.hstack((X, aaa))

    aaa = numpy.vstack((numpy.diff(diff, axis=0), numpy.array([[0, 0, 0], [0, 0, 0]])))

    X = numpy.hstack((X, aaa))

    #X[:-1,:-1] = X[1:, 0:3] - X[:-1, 0:3]
    scaler = MinMaxScaler(feature_range=(0, 1))
    xmin = X.min()
    xmax = X.max()
    scaler.fit(numpy.array([
        [xmin, xmin, xmin, xmin / 14, xmin / 14, xmin / 14, xmin / 14, xmin / 14, xmin / 14],
        [xmax, xmax, xmax, xmax / 14, xmax / 14, xmax / 14, xmax / 14, xmax / 14, xmax / 14]
    ]))
    rX = scaler.transform(X)
    # summarize transformed data
    numpy.set_printoptions(precision=3)


    for dim in range(0, 1):
        out = Image.new("RGB", (w, s), (0, 0, 0))

        # get a drawing context
        d = ImageDraw.Draw(out)

        for j in range(1, len(rX)):
            
            #print(rX[i])\
            row = rX[j]
            m = j
            x, y = (row[dim] * s % s), (row[(dim + 6)] * s) % s
            uu = out.getpixel((x, y))
            d.point((x, y), fill=(255 - m * 5 // 6, uu[1], uu[2]))
            uu = out.getpixel(((row[dim+1] * s % s), (row[(dim + 7)] * s) % s))

            d.point((row[dim+1] * s, row[(dim + 7)] * s), fill=(uu[0], 255 - m * 5 // 6, uu[2]))
            uu = out.getpixel(((row[dim+2] * s % s), (row[(dim + 8)] * s) % s))

            d.point((row[dim+2] * s, row[(dim + 8)] * s), fill=(uu[0], uu[1], 255 - m * 5 // 6))

        # draw multiline text
        #out.show()
        out.save('pictures/'+category+'_'+file+'_'+str(dim)+'.png', "PNG")

    #fig = pyplot.figure()
    #ax = Axes3D(fig)

    #sequence_containing_x_vals = rX[:,0]
    #sequence_containing_y_vals = rX[:,1]
    #sequence_containing_z_vals = rX[:,2]

    #ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    #pyplot.show()


im_x = []
im_y = []

for im_path in glob.glob("pictures/*.png"):
    im = imageio.imread(im_path)
    img_name = im_path.split('/')[1].split('.')[0]
    img_dim = img_name.split('_')[2]
    img_cat = img_name.split('_')[0]
    if img_dim == '0':
        im_x.append(im[:, :])
        im_y.append(int(img_cat) - 1)
    # do whatever wit

im_x = numpy.array(im_x)
im_x = im_x.reshape((im_x.shape[0], w, h, 3))

x_train = im_x[:2500].astype('float32')
x_test = im_x[2500:].astype('float32')
x_train /= 255
x_test /= 255

num_classes = 8
y_train = keras.utils.np_utils.to_categorical(im_y[:2500], num_classes)
y_test = keras.utils.np_utils.to_categorical(im_y[2500:], num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(w, h, 3)))
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=8,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
