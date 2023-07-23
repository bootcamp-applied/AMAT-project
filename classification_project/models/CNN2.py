from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model as tf_load_model
import joblib

# asaf net
class CNN2:
    def __init__(self, num_classes=10):
        self.batch_size = 32
        self.num_classes = num_classes
        self.epochs = 50
        self.input_shape = (32, 32, 3)

        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), padding='same',
                              input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, kernel_regularizer=l2(0.01)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

    def train(self, x_train, y_train, x_val, y_val):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

        self.history = self.model.fit(x_train, y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=(x_val, y_val),
                                      shuffle=True)
        return self.history

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate_accuracy(self, x_test, y_test):
        _, accuracy = self.model.evaluate(x_test, y_test)
        return accuracy

    def summary(self):
        self.model.summary()

    def save_model(self, filepath):
        self.model.save(filepath)

    def save_history(self, filepath):
        joblib.dump(self.history.history, filepath)

    @classmethod
    def load_cnn_model(cls, filepath):
        # Load a pre-trained model from a file and create a new instance of the class
        model = tf_load_model(filepath)
        loaded_cnn = cls()
        loaded_cnn.model = model
        return loaded_cnn

    @classmethod
    def load_cnn_history(cls, filepath):
        # Load training history (results) from a file and create a new instance of the class
        loaded_cnn = cls()
        loaded_cnn.history = joblib.load(filepath)
        return loaded_cnn
