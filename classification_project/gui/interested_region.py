import dash
from dash import Dash, dcc, html, Input, Output
from classification_project.utils.handling_new_image import NewImage
from PIL import Image
import base64
from io import BytesIO

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='image-graph',
        figure={
            'data': [],
            'layout': {
                'xaxis': {'range': [0, 400]},
                'yaxis': {'range': [0, 400]},
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': 0,
                    'y': 400,
                    'sizex': 400,
                    'sizey': 400,
                    'sizing': 'stretch',
                    'layer': 'below',
                    'source': im,
                }],
            },
        },
        config={'editable': True},
    ),
    html.Div([
        html.Div("Start: None", id="start"),
        html.Div("Stop: None", id="stop"),
        html.Div("Box: None", id="box"),
    ])
])
x0, y0 = None, None
x1, y1 = None, None
colors = ['blue', 'white']
index = False
figure = None
@app.callback(
    Output('start', 'children'),
    Output('stop', 'children'),
    Output('box', 'children'),
    Input('image-graph', 'relayoutData')
)
def update(relayoutData):
    global x0, y0, x1, y1, figure, index
    if 'xaxis.range[0]' in relayoutData:
        x0 = relayoutData['xaxis.range[0]']
    if 'yaxis.range[0]' in relayoutData:
        y0 = relayoutData['yaxis.range[0]']
    if 'xaxis.range[1]' in relayoutData:
        x1 = relayoutData['xaxis.range[1]']
    if 'yaxis.range[1]' in relayoutData:
        y1 = relayoutData['yaxis.range[1]']
    if figure:
        figure = dict(figure, shapes=[])
    if None not in (x0, y0, x1, y1):
        figure['shapes'] = [
            {
                'type': 'rect',
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1,
                'line': {'color': colors[index]},
            }
        ]
        index = not index
    return f'Start: ({x0}, {y0})', f'Stop: ({x1}, {y1})', f'Box: ({abs(x1-x0+1)}, {abs(y1-y0+1)})'
if __name__ == '__main__':
    app.run_server(debug=True)