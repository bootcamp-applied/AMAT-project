import pandas as pd
from preprocessing import PreProcessing
import tensorflow as tf

df = pd.read_csv('../DAL/cifar-100-df.csv')
preProcessing = PreProcessing(df)
preProcessing.preparing_data()
X_train, y_train, X_val, y_val, X_test, y_test = preProcessing.split_to_train_test_validation()

cifar_model = tf.keras.models.Sequential()

# First Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Second Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Max Pooling Layer
cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Third Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Fourth Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Max Pooling Layer
cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Fifth Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Sixth Layer
cifar_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
cifar_model.add(tf.keras.layers.BatchNormalization())

# Max Pooling Layer
cifar_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Flattening Layer
cifar_model.add(tf.keras.layers.Flatten())

# Dropout Layer
cifar_model.add(tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None))

# Adding the first fully connected layer
cifar_model.add(tf.keras.layers.Dense(units=256, activation='relu'))

# Output Layer
cifar_model.add(tf.keras.layers.Dense(units=20, activation='softmax'))

print(cifar_model.summary())

cifar_model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
cifar_model.fit(X_train, y_train, epochs=15)

test_loss, test_accuracy = cifar_model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
