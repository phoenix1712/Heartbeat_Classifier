from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
import data_preprocessor as dataimport
from keras.models import model_from_json
import argparse
import matplotlib.pyplot as plt
from keras.applications import VGG16

def train(normal_folder='data/Normal',
          abnormal_folder='data/Abnormal',
          max_features=5000,
          maxlen=100,
          batch_size=32,
          embedding_dims=100,
          nb_filter=90,
          hidden_dims=256,
          nb_epoch=500,
          nb_classes=2,
          optimizer='rmsprop',
          loss='categorical_crossentropy',
          test_split=0.2,
          seed=1955,
          model_json='hb_model_orthogonal_experiment_norm.json',
          weights='hb_weights_orthogonal_experiment_norm.hdf5',
          load_weights=False,
          normal_path='',
          abnormal_path=''):
    print('Loading data...')
    # Loads training data from specified folders for normal and abnormal sound files.
    # Transforms data using short-time Fourier transform, log scales the result and splits it into 129x129 squares.
    # Randomizes data.
    # Splits the data into train and test arrays
    (X_train, y_train), (X_test, y_test) = dataimport.load_data(normal_path=normal_path, abnormal_path=abnormal_path,
                                                                test_split=0.1, width=129, height=256, seed=seed,
                                                                category_split_equal=True)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    print('Build model...')
    model = Sequential()

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(129, 129, 3))
    model.add(conv_base)
    conv_base.trainable = False
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    # # We start off with using Convolution2D for a frame
    # # The filter is 3x57
    # model.add(Conv2D(nb_filter,
    #                  (3, 57),
    #                  padding='valid',
    #                  kernel_initializer='orthogonal',
    #                  kernel_regularizer=l2(0.0001),
    #                  kernel_constraint=maxnorm(2),
    #                  activation='relu',
    #                  input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    #
    # # we use standard max pooling (halving the output of the previous layer):
    # model.add(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))
    #
    # # the second convolution layer is 1x3
    # model.add(Conv2D(nb_filter,
    #                  (1, 3),
    #                  kernel_initializer='orthogonal',
    #                  kernel_regularizer=l2(0.0001),
    #                  kernel_constraint=maxnorm(2),
    #                  activation='relu'))
    #
    # # we use max pooling again:
    # model.add(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))
    #
    # # We flatten the output of the conv layer,
    # # so that we can add a vanilla dense layer:
    # model.add(Flatten())
    #
    # # we add two hidden layers:
    # # increasing number of hidden layers may increase the accuracy, current number is designed for the competition
    # model.add(Dense(512,
    #                 kernel_initializer='orthogonal',
    #                 kernel_regularizer=l2(0.0001),
    #                 kernel_constraint=maxnorm(2),
    #                 activation='relu'))
    # model.add(Dropout(0.5))
    #
    # # We project onto a binary output layer to determine the category (Currently: normal/abnormal,
    # # but you can try train on the exact abnormality also)
    # model.add(Dense(nb_classes, activation='softmax'))
    #
    # # you can load pre-trained weights to quicken the training
    # if load_weights:
    #     model.load_weights(weights)
    #
    # # Prints summary of the model
    # model.summary()
    #
    # # Compile the model
    # model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Saving model to Json (its easier to test it this way)
    json_string = model.to_json()
    open(model_json, 'w').write(json_string)

    # Each time the loss will drop it will save weights file
    checkpointer = ModelCheckpoint(filepath=weights, monitor='val_acc', verbose=1, save_best_only=True)

    # Start training
    history = model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              callbacks=[checkpointer],
              validation_data=(X_test, Y_test))

    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(acc_values)+1)

    plt.plot(epochs, acc_values,'bo',label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show();
    return True


def write_answer(filename, result, resultfile="answers.txt"):
    fo = open(resultfile, 'a')
    fo.write(str(filename) + "," + str(result) + "\n")
    fo.close()

    return True


def test(filename, model_json='hb_model_orthogonal_experiment_norm.json',
         weights='hb_weights_orthogonal_experiment_norm.hdf5',
         optimizer='sgd',
         loss='categorical_crossentropy',
         evaluate=False,
         model=Sequential()):

    if not evaluate:
        # Loads model from Json file
        model = model_from_json(open(model_json).read())

        # Loads pre-trained weights for the model
        model.load_weights(weights)

        # Compiles the model
        model.compile(loss=loss, optimizer=RMSprop(lr=2e-5))

    # loads filename, transfrorms data using short-time Fourier transform,
    # logscales the result and splits it into 129x129 squares
    X = dataimport.data_from_file(filename=str(filename) + ".wav", width=129, height=256, max_frames=10)

    predictions = np.zeros(len(X))
    z = 0

    # Makes predictions for each 129x129 square
    for frame in X:
        #predict_frame = np.zeros((1, 3, 129, 129))
        predict_frame = np.zeros((1, 129, 129, 3))
        predict_frame[0] = frame
        predictions_all = model.predict_proba(predict_frame, batch_size=batch_size)
        predictions[z] = predictions_all[0][1]

        z += 1

    # Averages the results of per-frame predictions
    average = np.average(predictions)
    average_prediction = round(average)

    # Prints the result
    if int(average_prediction) == 0.0:
        # append file with -1
        #write_answer(filename=filename, result="-1")
        return -1
        print('Result for ' + filename + ': ' + 'Normal (-1)')

    else:
        # append file with 1
        #write_answer(filename=filename, result="1")
        return 1
        print('Result for ' + filename + ': ' + 'Abnormal (1)')

    return int(average_prediction)


parser = argparse.ArgumentParser(description='This is a script to train and test PhysioNet 2016 challenge data.')
parser.add_argument('-o', '--option', help='Options may be train or test', required=True)
parser.add_argument('-i', '--inputfile', help='Input file name (full path) without .wav ending', required=False)

args = parser.parse_args()
# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 90
hidden_dims = 256
nb_epoch = 100
nb_classes = 2
sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
loss = 'binary_crossentropy'
model_json = 'hb_model_orthogonal_experiment_norm_vgg16_adam.json'
weights = 'hb_weights_orthogonal_experiment_norm_vgg16_adam.hdf5'
seed = 1995
test_split = 0.2
normal_path = 'data/Normal/'
abnormal_path = 'data/Abnormal/'

if (args.option == 'train'):
    train(max_features=max_features,
          maxlen=maxlen,
          batch_size=batch_size,
          embedding_dims=embedding_dims,
          nb_filter=nb_filter,
          hidden_dims=hidden_dims,
          nb_epoch=nb_epoch,
          nb_classes=nb_classes,
          optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5),
          loss=loss,
          test_split=test_split,
          seed=seed,
          model_json=model_json,
          weights=weights,
          load_weights=False,
          normal_path=normal_path,
          abnormal_path=abnormal_path)
elif (args.option == 'test'):
    test(model_json=model_json,
         weights=weights,
         optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5),
         loss=loss,
         filename=str(args.inputfile))
elif (args.option == 'evaluate'):
    testSet = pd.read_csv('test/REFERENCE.csv', header=None)
    count = 0
    # Loads model from Json file
    model = model_from_json(open(model_json).read())
    # Loads pre-trained weights for the model
    model.load_weights(weights)
    # Compiles the model
    model.compile(loss=loss, optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    for index, row in testSet.iterrows():
        #print("Index : ", index)
        pred = test(model_json=model_json,
                     weights=weights,
                     optimizer=sgd,
                     loss=loss,
                     filename='test/' + str(row[0]),
                     evaluate= True,
                     model = model)
        if pred == row[1]:
            count += 1
            #print('correct  :  ', count)
    print(count/301)
else:
    print('You need to choose between train and test arguments')