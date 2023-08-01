import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import classification_project.gui.gui_utils as gui_utils

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
        html.Div(id='similar-images-container'),
        # html.Div(
        #     [
        #         html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
        #                  style={'width': '300px', 'height': '300px'})
        #         for img_vector in similar_images
        #     ],
        #     style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}
        # ),
        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'})
    ]
)


@app.callback(
    [Output(component_id='image-label', component_property='children'),
     Output(component_id='uploaded-image', component_property='src'),
     Output(component_id='similar-images-container', component_property='children')],
    [Input(component_id='upload-data', component_property='contents')]
)
def classify_label(contents):
    label_mes = ''
    similar_images_container = None
    if contents is not None:
        formated_image = gui_utils.format_image(contents)
        label = gui_utils.predict_label(formated_image)
        label_mes = f'Your image label is: {label}'
        closest_images = gui_utils.similar_images()
        similar_images_container = [
            html.Div([
                html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
                         style={'width': '300px', 'height': '300px'})
            ]) for img_vector in closest_images
        ]
    return label_mes, contents, similar_images_container


if __name__ == '__main__':
    app.run_server(debug=True)
