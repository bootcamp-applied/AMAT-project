from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, ReLU, Add, Input
from tensorflow.keras.models import Model
from classification_project.utils.load_data import load_data

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def resnet_block(inputs, filters, strides=1, use_conv_shortcut=False):
    shortcut = inputs
    if use_conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, 3, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = resnet_block(x, filters=64, use_conv_shortcut=True)
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)

    x = resnet_block(x, filters=128, strides=2, use_conv_shortcut=True)
    x = resnet_block(x, filters=128)
    x = resnet_block(x, filters=128)

    x = resnet_block(x, filters=256, strides=2, use_conv_shortcut=True)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=256)

    x = resnet_block(x, filters=512, strides=2, use_conv_shortcut=True)
    x = resnet_block(x, filters=512)
    x = resnet_block(x, filters=512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='resnet18')
    return model


# Load CIFAR-10 dataset and preprocess
x_train, y_train, x_test, y_test = load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=15)
y_test = to_categorical(y_test, num_classes=15)

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# CIFAR-10 dataset parameters
input_shape = (32, 32, 3)
num_classes = 15

# Build the model
model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test accuracy: {test_accuracy:.4f}")

model.save('../saved_model/res_net_18_model.h5')
