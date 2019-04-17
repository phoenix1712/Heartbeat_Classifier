from __future__ import print_function
import numpy as np
import data_preprocessor as dataimport
from keras.models import Sequential

batch_size = 32


def test(filename,
         model = Sequential()):
    print('Build model...')

    # loads filename, transforms data using short-time Fourier transform, log scales the result and splits
    # it into 129x129 squares
    X = dataimport.data_from_file(filename=str(filename), width=129, height=256, max_frames=10)

    predictions = np.zeros(len(X))
    z = 0

    # Makes predictions for each 129x129 square
    for frame in X:
        predict_frame = np.zeros((1, 3, 129, 129))
        predict_frame[0] = frame
        predictions_all = model.predict_proba(predict_frame, batch_size=batch_size)
        predictions[z] = predictions_all[0][1]

        z += 1

    print(predictions)
    # Averages the results of per-frame predictions
    average = np.average(predictions)
    average_prediction = round(average)

    return int(average_prediction)

