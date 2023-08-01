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
        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'})
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


if __name__ == '__main__':
    app.run_server(debug=True)
