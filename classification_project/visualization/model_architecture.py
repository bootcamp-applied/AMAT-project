from tensorflow.keras.models import load_model as load_model
import visualkeras

# tf_load_model('/content/drive/MyDrive/saved_cnn_model.h5')
loaded_model = load_model('../saved_model/saved_cnn_model.h5')
visualkeras.layered_view(loaded_model, to_file='model.png', legend=True)
loaded_model.summary()