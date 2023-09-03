import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# Load your CNN model here
cnn_model = load_model('path_to_your_model.h5')  # Update with your model's path

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Camera App"),
    html.Button("Open Camera", id="open-camera"),
    html.Button("Capture Image", id="capture-image"),
    html.Button("Save Image", id="save-image"),
    dcc.Graph(id="live-feed"),
])

@app.callback(
    Output("live-feed", "figure"),
    Input("open-camera", "n_clicks"),
    State("live-feed", "relayoutData"),
)
def open_camera(n_clicks, relayout_data):
    global camera
    if n_clicks is None:
        return dash.no_update

    frame = camera.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encoded_image = cv2.imencode('.png', frame)[1].tobytes()

    figure = {
        'data': [{'x': [], 'y': [], 'type': 'scattergl', 'mode': 'lines+markers', 'marker': {'size': 0.1}}],
        'layout': {'xaxis': {'range': [0, 1]}, 'yaxis': {'range': [0, 1]}, 'height': 400}
    }

    return {
        'data': [{
            'x': [0, 1],
            'y': [0, 1],
            'mode': 'markers',
            'marker': {'size': 1, 'opacity': 0},
            'image': encoded_image,
        }],
        'layout': figure['layout']
    }

@app.callback(
    Output("live-feed", "figure"),
    Input("capture-image", "n_clicks"),
    State("live-feed", "figure"),
)
def capture_image(n_clicks, figure):
    global camera
    if n_clicks is None:
        return dash.no_update

    frame = camera.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encoded_image = cv2.imencode('.png', frame)[1].tobytes()

    return {
        'data': [{
            'x': [0, 1],
            'y': [0, 1],
            'mode': 'markers',
            'marker': {'size': 1, 'opacity': 0},
            'image': encoded_image,
        }],
        'layout': figure['layout']
    }

@app.callback(
    Output("live-feed", "figure"),
    Input("save-image", "n_clicks"),
    State("live-feed", "figure"),
)
def save_image(n_clicks, figure):
    if n_clicks is None:
        return dash.no_update

    image_bytes = figure['data'][0]['image']
    image = Image.open(BytesIO(image_bytes))
    image.save("captured_image.jpg")

    return dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True)