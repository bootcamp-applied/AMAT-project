# Import necessary libraries
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import base64
from PIL import Image, ImageOps
import io
import numpy as np
import matplotlib.pyplot as plt
from classification_project.models.CNN1 import CNN1
from classification_project.models.CNN2 import CNN2
import json

def convert_to_category(label):
    map_label = '../../classification_project/utils/dict.json'
    with open(map_label, 'r') as f:
        label_dict = json.load(f)
    label_category = next((val for key, val in label_dict.items() if key == str(label)), None)
    return label_category


# Define your image classification function (replace this with your actual function)
def get_image_label(image_array):
    resized_image = Image.fromarray(image_array).resize((32, 32), Image.BICUBIC)
    plt.imshow(resized_image)
    plt.show()

    # resized_image = Image.fromarray(image_array)
    # resized_image = ImageOps.exif_transpose(resized_image)  # Handle EXIF orientation (optional)
    # resized_image = resized_image.resize((32, 32), Image.BILINEAR, reducing_gap=3.0)

    loaded_model = CNN2.load_cnn_model('../../classification_project/saved_model/saved_cnn_model_2.keras').model
    input_image = np.expand_dims(resized_image, axis=0)
    probabilities = loaded_model.predict(input_image)
    # probabilities = loaded_model.predict(np.array(resized_image).shape)
    label = np.argmax(probabilities)
    label_category = convert_to_category(label)
    return label_category

    return "cat"

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'center',
        'justify-content': 'center',
        'height': '100vh'
    },
    children=[
        html.H1("Welcome to our Image Classifier", style={'text-align': 'center', 'margin-bottom': '20px'}),
        dcc.Upload(id='upload-data', children=html.Button('Upload Image'), style={'margin-bottom': '20px'}),

        html.Div(
            style={'text-align': 'center'},
            children=[
                html.Img(id='uploaded-image', src='', style={'width': '300px'})
            ]
        ),
        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'})
    ]
)

@app.callback(
    [Output(component_id='image-label', component_property='children'),
     Output(component_id='uploaded-image', component_property='src')],
    [Input(component_id='upload-data', component_property='contents')]
)
def classify_label(contents):
    label = 'Unknown'
    content_string = None
    if contents is not None:
        # Decode the uploaded image data
        _, content_string = contents.split(',')
        decoded_image = base64.b64decode(content_string)

        # Convert the bytes object to a NumPy array using PIL
        img = Image.open(io.BytesIO(decoded_image))
        image_array = np.array(img)

        # If image has 4 channels (RGBA), remove alpha channel
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]  # Select only the first 3 channels (RGB)

        # Now you have the image as a 3-channel (RGB) NumPy array, pass it to your classification function
        label = get_image_label(image_array)

    return f'Your image label is: {label}', f"data:image/png;base64,{content_string}"


if __name__ == '__main__':
    app.run_server(debug=True)
