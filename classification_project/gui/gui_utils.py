def similar_images_CNN_features():
    global new_image
    # Load the DataFrame with flattened images
    data_features = pd.read_feather('../../data/processed/features_after_CNN.feather')
    data = pd.read_feather('../../data/processed/cifar_10_100.feather')
    data = data.iloc[:, 2:]
    # Preprocess the images
    reshaped_image_array = new_image.reshape(1, 32, 32, 3)
    model = CNN.load_cnn_model('../../classification_project/saved_model/saved_cnn_model.h5').model
    preprocessed_image = preprocess_input(reshaped_image_array)
    feature_layer_index = -2  # Index of the layer before the final dense layer
    feature_layer_output = model.layers[feature_layer_index].output
    # Create a Keras function to extract features from the given input images
    get_features = K.function([model.input], [feature_layer_output])
    new_image_features = get_features([preprocessed_image])[0]
    similarity_scores = cosine_similarity(new_image_features, data_features)
    four_closest_indices = np.argsort(similarity_scores[0])[::-1][:4]
    # te read data features
    four_closest_vectors = data.iloc[four_closest_indices]
    return four_closest_vectors
