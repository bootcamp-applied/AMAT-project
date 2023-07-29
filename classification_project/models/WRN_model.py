# import tensorflow as tf
# from classification_project.utils.load_data import load_data
# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Add, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.utils import to_categorical
#
# # Load CIFAR-10-100 dataset
# x_train, y_train, x_test, y_test = load_data()
# y_train = to_categorical(y_train, num_classes=15)
# y_test = to_categorical(y_test, num_classes=15)
#
# # Wide ResNet model definition
# def wide_resnet(depth, widen_factor, num_classes=15):
#     def basic_block(x, filters, strides=1):
#         x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         return x
#
#     def residual_block(x, filters, strides=1):
#         shortcut = x
#
#         # If the stride is not equal to 1 or the number of filters has changed,
#         # use a 1x1 convolution to match the dimensions
#         if strides != 1 or x.shape[-1] != filters:
#             shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
#             shortcut = BatchNormalization()(shortcut)
#
#         x = basic_block(x, filters, strides)
#         x = basic_block(x, filters)
#
#         # Update the following line to match the dimensions with the correct number of filters
#         if x.shape[-1] != filters:
#             shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
#             shortcut = BatchNormalization()(shortcut)
#
#         x = Add()([x, shortcut])
#         return x
#
#
#
#     n = (depth - 4) // 6
#     k = widen_factor
#
#     inputs = Input(shape=(32, 32, 3))
#     x = Conv2D(16, kernel_size=3, strides=1, padding='same')(inputs)
#
#     for i in range(3):
#         x = residual_block(x, 16*k)
#     for i in range(3):
#         x = residual_block(x, 32*k, strides=2)
#     for i in range(3):
#         x = residual_block(x, 64*k, strides=2)
#
#     x = AveragePooling2D(pool_size=8)(x)
#     x = Flatten()(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs, outputs)
#     return model
#
# # Create the model
# model = wide_resnet(depth=28, widen_factor=10)
#
# # Compile the model
# optimizer = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
#
# # Evaluate the model
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f'Test accuracy: {test_acc:.2f}%')
#
#
# # Save the model to a file
# model.save('../saved_model/wide_resnet_model.h5')








