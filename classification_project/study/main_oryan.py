
from classification_project.study.use_Visualization import plot_images_to_given_label
if __name__ == '__main__':
    plot_images_to_given_label('airplane')
    loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
    # plot_model(loaded_model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    visualkeras.layered_view(loaded_model.model)