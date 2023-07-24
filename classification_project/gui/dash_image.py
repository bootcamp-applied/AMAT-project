
import base64
from classification_project.utils.handling_new_image import NewImage
import dash
import numpy as np
from dash import html
from dash import dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

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
    # if contents is not None:
    #     format_image(contents)
    label = 'cat'
    label_mes = f'Your image label is: {label}'
    return label_mes, contents


def format_image(image):
    content_type, content_string = image.split(',')
    decoded_img = base64.b64decode(content_string)
    # Convert decoded image data to numpy array
    np_arr = np.frombuffer(decoded_img, np.uint8)
    handler = NewImage()
    handler.image_handle(np_arr)
    print(np_arr)


if __name__ == '__main__':
    app.run_server(debug=True)