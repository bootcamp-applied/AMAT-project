# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from classification_project.preprocessing.preprocessing import Preprocessing
# import numpy as np
# import pandas as pd
#
#  # Load pre-trained MobileNet model
# loaded_mobilenet = MobileNet(weights='imagenet', include_top=True)
# #
# # # Modify the output layer for your new number of classes
# # new_num_classes = 15  # CIFAR-10 + 5 superclasses from CIFAR-100
# # loaded_mobilenet.layers.pop()  # Remove the last layer
# # loaded_mobilenet.layers[-1].outbound_nodes = []  # Disconnect the previous output layer
# # loaded_mobilenet.add(Dense(new_num_classes, activation='softmax'))  # Add new output layer
# #
# # # Compile the model
# # loaded_mobilenet.compile(
# #     loss='categorical_crossentropy',
# #     optimizer=Adam(learning_rate=0.001),  # Adjust the learning rate as needed
# #     metrics=['accuracy']
# # )
# new_num_classes = 15  # CIFAR-10 + 5 superclasses from CIFAR-100
# loaded_mobilenet.layers.pop()  # Remove the last layer
#
# # Add a new dense output layer
# output_layer = Dense(new_num_classes, activation='softmax', name='new_output')(loaded_mobilenet.layers[-1].output)
#
# # Create a new model with the modified output layer
# modified_mobilenet = Model(inputs=loaded_mobilenet.input, outputs=output_layer)
#
# # Compile the model
# modified_mobilenet.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(),  # Use an appropriate optimizer
#     metrics=['accuracy']
# )
#
# # Load and preprocess your new dataset (replace with your data loading code)
# # new_x_train, new_y_train, new_x_val, new_y_val = ...
# df = pd.read_csv('data/processed/cifar-10-100-augmentation.csv')
# preprocessing = Preprocessing(df)
# preprocessing.prepare_data()
# x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)
#
# # Train the model while keeping the earlier layers frozen
# # epochs = 20  # Adjust as needed
# # history = loaded_mobilenet.fit(
# #     x_train, y_train,
# #     batch_size=32,
# #     epochs=epochs,
# #     validation_data=(x_val, y_val),
# #     shuffle=True
# # )
# epochs = 20  # Adjust as needed
# history = modified_mobilenet.fit(
#     x_train, y_train,
#     batch_size=32,
#     epochs=epochs,
#     validation_data=(x_val, y_val),
#     shuffle=True
# )
#
# # Evaluate the model on the test set
# test_loss, test_accuracy = loaded_mobilenet.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_accuracy:.4f}")


from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from classification_project.preprocessing.preprocessing import Preprocessing
import numpy as np
from tensorflow.keras.models import Model
import pandas as pd

# Load pre-trained MobileNet model
loaded_mobilenet = MobileNet(weights='imagenet', include_top=True)

# Modify the output layer for your new number of classes
new_num_classes = 15  # CIFAR-10 + 5 superclasses from CIFAR-100
loaded_mobilenet.layers.pop()  # Remove the last layer

# Add a new dense output layer
output_layer = Dense(new_num_classes, activation='softmax', name='new_output')(loaded_mobilenet.layers[-1].output)

# Create a new model with the modified output layer
modified_mobilenet = Model(inputs=loaded_mobilenet.input, outputs=output_layer)

# Compile the model
modified_mobilenet.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),  # Use an appropriate optimizer
    metrics=['accuracy']
)

# Load and preprocess your new dataset (replace with your data loading code)
df = pd.read_csv("../../data/processed/cifar-10-100-augmentation.csv")
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

#Train the model while keeping the earlier layers frozen
epochs = 20  # Adjust as needed
history = modified_mobilenet.fit(
    x_train, y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(x_val, y_val),
    shuffle=True
)

#Evaluate the model on the test set
test_loss, test_accuracy = modified_mobilenet.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
