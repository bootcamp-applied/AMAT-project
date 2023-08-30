import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State, ALL
import classification_project.gui.gui_utils as gui_utils


# Initialize the Dash app
app = dash.Dash(__name__)
confirmation_section = html.Div(
    id='confirmation-section',
    style={'text-align': 'center', 'margin-top': '20px'},
    children=[
        html.H2("Let us know if the label is right"),
        html.Button("Yes", id='confirm-yes', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("No", id='confirm-no', n_clicks=0)
    ]
)
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
            id='uploaded-image-container',
            style={'text-align': 'center', 'border': '2px solid black'},
            children=[
                dcc.Graph(id='uploaded-image', config={'doubleClick': 'reset'})
            ]
        ),
        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'}),
        confirmation_section,
        html.Div(id='similar-images-heading', children=[
            html.H2("Give us another chance", style={'text-align': 'center', 'margin-top': '20px'}),
            html.H2("Explore similar images from our training set. Click on the image with the same label"
                    , style={'text-align': 'right', 'margin-bottom': '20px','font-size': '18px'})
        ]),
        html.Div(
            id='similar-images-container',
            style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}
        )
    ]
)

# Create figure
img_width = 600
img_height = 400
scale_factor = 0.5

fig = go.Figure()

# Add invisible scatter trace.
fig.add_trace(
    go.Scatter(
        x=[0, img_width * scale_factor],
        y=[0, img_height * scale_factor],
        mode="markers",
        marker_opacity=0
    )
)

# Configure axes
fig.update_xaxes(
    visible=False,
    range=[0, img_width * scale_factor]
)

fig.update_yaxes(
    visible=False,
    range=[0, img_height * scale_factor],
    scaleanchor="x"
)

# Configure other layout
fig.update_layout(
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)


@app.callback(
    [Output(component_id='image-label', component_property='children'),
     Output(component_id='uploaded-image', component_property='data')],
    [Input(component_id='upload-data', component_property='contents')]
)
def classify_label(contents):
    label_mes = ''
    if contents is not None:
        formated_image = gui_utils.format_image(contents)
        label = gui_utils.predict_label(formated_image)
        label_mes = f'Your image label is: {label}'
    return label_mes, contents

# Callback to update uploaded image
@app.callback(
    Output('uploaded-image', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_uploaded_image(contents, filename):
    if contents is None:
        return fig

    # Process the uploaded image
    # You can use your image processing code here to crop and display the image
    # For now, let's just use the uploaded image without any processing
    encoded_image = contents.split(',')[1]

    # Update the figure with the uploaded image
    fig.update_layout(
        images=[
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=f"data:image/png;base64,{encoded_image}"
            )
        ]
    )

    return fig

def create_image_div(similarity_phrase, img_vector, index):
    return html.Div([
        similarity_phrase,
        html.A(
            html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
                     style={'width': '150px', 'height': '150px', 'border': '2px solid black'}),
            href='',  # Leave href empty
            id={'type': 'image-click', 'index': index},  # Include an id with the image index
        )
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'padding': '15px'})

# Callback to update similar images
@app.callback(
    [Output('similar-images-heading', 'style'),
     Output('similar-images-container', 'children')],
    [Input('confirm-no', 'n_clicks')]
)
def update_similar_images(n_clicks_no):
    similar_images_container = None
    heading_style = {'display': 'none'}

    if n_clicks_no is not None and n_clicks_no > 0:
        closest_images = gui_utils.similar_images_using_potential()
        similar_images_container = []

        ranking_phrases = ["Most Similar", "Second Most Similar", "Third Most Similar", "Fourth Most Similar"]

        for i, img_vector in enumerate(closest_images.values, start=0):
            if i >= len(ranking_phrases):
                break

            similarity_phrase = html.Div(ranking_phrases[i], style={'text-align': 'center', 'font-weight': 'bold'})
            image_div = create_image_div(similarity_phrase, img_vector, i)  # Create the image_div
            similar_images_container.append(image_div)

        heading_style = {'text-align': 'right', 'margin-bottom': '20px'}

    return heading_style, similar_images_container


# to ask if we were right only when it predicted
@app.callback(
    Output('confirmation-section', 'style'),
    [Input('image-label', 'children'),
     Input('confirm-yes', 'n_clicks'),
     Input('confirm-no', 'n_clicks')]
)
def toggle_confirmation_section(image_label, n_clicks_yes, n_clicks_no):
    if image_label and image_label.strip():
        if n_clicks_yes is not None and n_clicks_yes > 0:
            return {'display': 'none'}
        elif n_clicks_no is not None and n_clicks_no > 0:
            return {'display': 'none'}
        else:
            return {'text-align': 'center', 'margin-top': '20px'}
    else:
        return {'display': 'none'}

# to plot the similar images if "no" is pressed
@app.callback(
    Output('similar-images-container', 'style'),
    Input('confirm-no', 'n_clicks')
)
def toggle_similar_images_section(n_clicks_no):
    if n_clicks_no is not None and n_clicks_no > 0:
        return {'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('clicked-image-number', 'children'),  # Update a component to display clicked image number
    Input({'type': 'image-click', 'index': ALL}, 'n_clicks'),
    State('image-data', 'data')
)
def handle_image_clicks(n_clicks_list, image_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    clicked_image_index = ctx.triggered[0]['prop_id'].split('.')[0]['index']
    clicked_image_number = clicked_image_index + 1  # Since indices start from 0
    print(clicked_image_number)
    return f"You clicked on image number: {clicked_image_number}"


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)