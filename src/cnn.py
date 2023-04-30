# data tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

from tensorflow.keras.optimizers import SGD

def load_gafs(gaf_path):
    """Load gaf images from path and return them as a numpy array"""
    gafs = []
    labels = []

    for file in gaf_path.iterdir():
        if file.is_file():
            gaf = np.load(file)
            gafs.append(gaf)
            
            label = str(file).split('.npy')[0]
            label = label.split('_')[-1]
            if label == 'A' or label == 'F': # if participant suffered from alzheimers or frontotemporal dementia
                labels.append(1)
            else:
                labels.append(0)
   
    gafs = np.array(gafs)
    labels = np.array(labels)
    
    return gafs, labels

def prep_model():
    # Initalise model
    model = Sequential()

    # Define CONV => ReLU
    model.add(Conv3D(32, 
                    kernel_size = 6,
                    padding = "same",
                    input_shape = (38, 38, 19, 3),
                    kernel_regularizer='l1'))
    
    model.add(Activation("relu"))

    # Define CONV => ReLU
    """model.add(Conv3D(32, 
                    kernel_size = 3,
                    padding = "same", 
                    kernel_regularizer='l1'))
    
    model.add(Activation("relu"))
    """
    # FC classifier
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(3))
    model.add(Activation("softmax"))

    # compile model
    opt = SGD(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", 
                optimizer=opt,
                metrics=["accuracy"])

    return model

def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train,
                        validation_split = 0.2,
                        batch_size=64,
                        epochs=3,
                        verbose=1)

    return model, history

def test_model(model, X_test, y_test, label_names):
    # evaluate model
    predictions = model.predict(X_test, batch_size = 32)

    clf_report = classification_report(y_test.argmax(axis=1),
                                       predictions.argmax(axis=1),
                                       target_names=label_names)
    
    return clf_report

def plot_history(history, save_path:Path=None):
    """
    Plots the training history.

    Parameters
    ----------
    history : History
        Training history.
    save_path : Path, optional
        Path to save the plot to, by default None
    
    Returns
    -------
    fig : Figure
        Figure object.
    axes : Axes
        Axes object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi = 300, sharex=True)

    # plot accuracy
    axes[0].plot(history.history["accuracy"], label='train')
    axes[0].plot(history.history["val_accuracy"],linestyle = "--",label="val")
    axes[0].set_title("Accuracy")

    # plot loss
    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val', linestyle = "--")
    axes[1].set_title('Loss')

    # add legend
    axes[0].legend()
    axes[1].legend()

    # add labels
    fig.supxlabel('Epoch')

    plt.tight_layout()

    # save plot
    if save_path:
        plt.savefig(save_path)
    
    return fig, axes

def balance_classes(X, y):
    """
    Balances the class weight by removing trials from classes with more trials

    Parameters
    ----------
    X : array
        Data array with n_trials as the first dimension
    y : array
        Array with shape (n_trials, )
    
    Returns
    -------
    X_equal : array
        Data array with a equal number of trials for each class
    y_equal : array
        Array with shape (n_trials, ) containing classes with equal number of trials for each class
    """
    keys, counts = np.unique(y, return_counts = True)

    # get the minimum number of trials
    min_count = np.min(counts)

    # loop through each class and remove trials
    remove_ind = []
    for key, count in zip(keys, counts):
        index = np.where(np.array(y) == key)
        random_choices = np.random.choice(len(index[0]), size = count-min_count, replace=False)
        remove_ind.extend([index[0][i] for i in random_choices])
    
    X_equal = np.delete(X, remove_ind, axis = 0)
    y_equal = np.delete(y, remove_ind, axis = 0)

    return X_equal, y_equal

def main():
    path = Path(__file__)
    gaf_path = path.parent.parent / 'data' / 'gaf'
    gafs, labels = load_gafs(gaf_path)

    gafs, labels = balance_classes(gafs, labels)

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(gafs, labels, test_size=0.2, stratify=labels, random_state=42)
    
    # convert labels from integers to vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # initialize model
    model = prep_model()

    # train model
    model, history = train_model(model, X_train, y_train)

    # evaluate model
    clf_report = test_model(model, X_test, y_test, label_names=["Alzheimers or Dementia", "Control"])

    report_path = path.parents[1] / 'mdl' / 'cnn.txt'

    with open(report_path, 'w') as f:
        f.write(clf_report)

    # save model
    model_path = path.parents[1] / 'mdl' / 'cnn.h5'
    model.save(model_path)

    # plot history
    plot_history(history, save_path =  path.parents[1] / 'mdl' / 'history.png')

if __name__ == '__main__':
    main()