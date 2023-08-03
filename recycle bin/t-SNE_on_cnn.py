import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from keras.models import Model

from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN

df = pd.read_csv('../data/processed/cifar-10-100.csv', dtype='int')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../classification_project/saved_model/saved_cnn_model.keras')
loaded_history_model = CNN.load_cnn_history('../classification_project/saved_model/saved_cnn_history.pkl')

feat_extractor = Model(inputs=loaded_model.model.input,
                       outputs=loaded_model.model.get_layer('dense').output)

features = feat_extractor.predict(x_test)

tsne = TSNE().fit_transform(features)
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100

# t-SNE with images
full_image = Image.new('RGB', (width, height))
for idx, x in enumerate(x_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       resample=Image.Resampling.LANCZOS)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))



plt.imshow(full_image)

# t-SNE with points

# y_pred = loaded_model.predict(x_test)
# features = y_pred.reshape((-1, np.prod(y_pred.shape[1:])))

y_test = np.argmax(y_test, axis = 1)
labels = np.reshape(y_test, (y_test.shape[0], 1))

tsne = TSNE(n_components=2, random_state=0)

test_representations_2d = tsne.fit_transform(features)

# Create a plot
plt.figure(figsize=(10,10))
scatter = plt.scatter(test_representations_2d[:, 0], test_representations_2d[:, 1], c=labels.flatten(), cmap='tab20') # s = 10

# Create a legend with class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish','people','flowers','trees','fruit and vegetables']
legend1 = plt.legend(*scatter.legend_elements(num=10), title="Classes")
plt.gca().add_artist(legend1)

# Convert labels to class names
class_indices = [int(label.get_text().split("{")[-1].split("}")[0]) for label in legend1.texts]
for t, class_index in zip(legend1.texts, class_indices):
    t.set_text(class_names[class_index])

plt.show()
print("DASfd")
print("DASfd")
