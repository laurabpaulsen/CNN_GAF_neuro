# data tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

from tensorflow.keras.optimizers import SGD


def load_gafs(path):
    """Load gaf images from path and return them as a numpy array"""
    gafs = []
    labels = []

    for file in os.listdir(path):
        if file.endswith('.npy'):
            gaf = np.load(os.path.join(path, file))
            gafs.append(gaf)
            label = file.split('.npy')[0][-1]
            labels.append(label)

    gafs = np.array(gafs)
    labels = np.array(labels)

    return gafs, labels

def prep_model():
    #initalise model
    model = Sequential()

    # define CONV => ReLU
    model.add(Conv2D(32, 
                    (3,3),
                    padding = "same",
                    input_shape = (32, 32, 3)))
    model.add(Activation("relu"))
            
    # FC classifier
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    # compile model
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", 
                optimizer=opt,
                metrics=["accuracy"])

    return model

def train_model(model, X_train, y_train):
    # train model
    history = model.fit(X_train, y_train,
                        validation_split = 0.1,
                        batch_size=32,
                        epochs=100,
                        verbose=1)

    return model, history

def test_model(model, X_test, y_test):
    # evaluate model
    predictions = model.predict(X_test, batch_size=32)

    clf_report = classification_report(y_test.argmax(axis=1),
                                       predictions.argmax(axis=1),
                                       target_names=label_names))

    return clf_report

def main():
    path = Path(__file__)
    gaf_path = path.parent.parent / 'data' / 'gaf'
    gafs, labels = load_gafs(gaf_path)

    # scale images to the range [0, 1]
    gafs = gafs.astype('float') / 255.0

    print(gafs.shape)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(gafs, labels, test_size=0.2, random_state=42)

    # convert labels from integers to vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # initialize model
    model = prep_model()

    # train model
    model, history = train_model(model, X_train, y_train)

    # evaluate model
    clf_report = test_model(model, X_test, y_test)
    report_path = path.parents[2] / 'mdl' / 'cnn.txt'

    with open(report_path, 'w') as f:
        f.write(clf_report)

    # save model
    model_path = path.parents[2] / 'mdl' / 'cnn.h5'
    model.save(model_path)







if __name__ == '__main__':
    main()