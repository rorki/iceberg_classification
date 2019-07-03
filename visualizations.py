import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def show_bands(data):
    band_im1 = np.reshape(data['band_1'], (75, 75))
    band_im2 = np.reshape(data['band_2'], (75, 75))

    fig = plt.figure()

    ax = plt.subplot(121)
    ax.set_title('Band 1: HH', fontsize=12)
    ax.imshow(band_im1)

    ax = plt.subplot(122)
    ax.set_title('Band 2: HV', fontsize=12)
    ax.imshow(band_im2)

    plt.show()

def plot_history(history):
    # history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, square=True, cmap='RdYlGn')
    plt.show()


def plot_cnn_model(model):
    plot_model(model, to_file='plots/model.png')
    img = mpimg.imread('plots/model.png')
    #fig = plt.figure()
    #fig.suptitle('CNN architecture', fontsize=12)
    plt.imshow(img)