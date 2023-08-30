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
        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'}),
        html.Div(id='similar-images-heading', children=[
            html.H2("Most similar images from training set", style={'text-align': 'right', 'margin-bottom': '20px'})
        ]),
        html.Div(
            id='similar-images-container',
            style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}
        )
    ]
)


@app.callback(
    [Output(component_id='image-label', component_property='children'),
     Output(component_id='uploaded-image', component_property='src')],
    [Input(component_id='upload-data', component_property='contents')]
)
def classify_label(contents):
    label_mes = ''
    if contents is not None:
        formated_image = gui_utils.format_image(contents)
        label = gui_utils.predict_label(formated_image)
        label_mes = f'Your image label is: {label}'
    return label_mes, contents


@app.callback(
    [Output(component_id='similar-images-heading', component_property='style'),
     Output(component_id='similar-images-container', component_property='children')],
    [Input(component_id='image-label', component_property='children')]
)
def update_similar_images(label_mes):
    similar_images_container = None
    heading_style = {'display': 'none'}

    if label_mes:
        closest_images = gui_utils.similar_images_using_potential()
        similar_images_container = []

        ranking_phrases = ["Most Similar", "Second Most Similar", "Third Most Similar", "Fourth Most Similar"]

        for i, img_vector in enumerate(closest_images.values, start=0):
            if i >= len(ranking_phrases):
                break

            similarity_phrase = html.Div(ranking_phrases[i], style={'text-align': 'center', 'font-weight': 'bold'})
            image_div = html.Div([
                similarity_phrase,
                html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
                         style={'width': '300px', 'height': '300px'})
            ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
            similar_images_container.append(image_div)

        heading_style = {'text-align': 'right', 'margin-bottom': '20px'}

    return heading_style, similar_images_container


if __name__ == '__main__':
    app.run_server(debug=True)