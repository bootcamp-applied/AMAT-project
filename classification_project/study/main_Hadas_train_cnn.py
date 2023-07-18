import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.visualization.visualization import confusion_matrix
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('../../data/processed/cifar-10.csv')
    preprocessing = Preprocessing(df)
    preprocessing.prepare_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.split_data()
    #
    cifar_model = tf.keras.models.Sequential()
    #
    # First Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Second Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Max Pooling Layer
    cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    #
    # Third Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Fourth Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Max Pooling Layer
    cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    #
    # Fifth Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Sixth Layer
    cifar_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    cifar_model.add(tf.keras.layers.BatchNormalization())
    #
    # Max Pooling Layer
    cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    #
    # Flattening Layer
    cifar_model.add(tf.keras.layers.Flatten())
    #
    # Dropout Layer
    cifar_model.add(tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None))
    #
    # Adding the first fully connected layer
    cifar_model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    #
    # Output Layer
    cifar_model.add(tf.keras.layers.Dense(units=20, activation='softmax'))
    #
    # print(cifar_model.summary())
    #
    cifar_model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    cifar_model.fit(X_train, y_train, epochs=15)
    #
    #test_loss, test_accuracy = cifar_model.evaluate(X_test, y_test)
    predictions = cifar_model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10', 'class11', 'class12', 'class13', 'class14']
    confusion_matrix(y_test, predicted_labels, class_names)
    # print("Test accuracy: {}".format(test_accuracy))
    print('hi')
