import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools


def load_data():
    # Load in the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape data for convolutional layer
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, y_train, x_test, y_test


def build_model(input_shape, num_classes):
    # Build the model using function API
    i = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=i, outputs=x)
    return model


def plot_loss_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()


def plot_accuracy_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_misclassified_images(x_test, y_test, p_test, labels):
    missclassified_idx = np.where(p_test != y_test)[0]
    i = np.random.choice(missclassified_idx)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('True label: %s Predicted label: %s' % (labels[y_test[i]], labels[p_test[i]]))
    plt.show()


def initializeAndDetect():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Numbers of classes
    K = len(set(y_train))
    print("number of classes: ", K)

    # Build model
    model = build_model(x_train[0].shape, K)

    # Compile and fit
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

    # Plot loss and accuracy
    plot_loss_history(r)
    plot_accuracy_history(r)

    # Evaluate model
    p_test = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))

    labels = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoots']
    plot_misclassified_images(x_test, y_test, p_test, labels)


if __name__ == "__main__":
    initializeAndDetect()
