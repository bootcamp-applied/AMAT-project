import base64

import numpy as np
import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State, ALL
import classification_project.gui.gui_utils as gui_utils
import matplotlib.pyplot as plt
import io
from PIL import Image

from functools import partial

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
        'height': '100vh',
        'background-color': 'gray',  # Set your desired background color here
        'overflow-y': 'auto'
},
    children=[
        html.H1("Welcome to our Image Classifier",
                style={'text-align': 'center', 'margin-bottom': '20px', 'display': 'block', 'font-size': '50px',
                       'color': 'white', 'text-shadow': '4px 4px 6px #000000'}),
        dcc.Upload(id='upload-data', children=html.Button('Upload Image'), style={'margin-bottom': '20px'}),

        html.Div(
            id='uploaded-image-container',
            style={'text-align': 'center', 'border': '4px solid black', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 1.5)'},
        children=[
                dcc.Graph(id='uploaded-image', config={'doubleClick': 'reset'})
            ]
        ),

        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'}),
        html.H2(id='label-after-crop', style={'text-align': 'center', 'margin-top': '20px'}),
        confirmation_section,
        html.Div(id='similar-images-heading', children=[
            html.H2("Give us another chance", style={'text-align': 'center', 'margin-top': '-20px'}),
            html.H2("Explore similar images from our training set. Click the button above the image with the same label",
                    style={'text-align': 'right', 'margin-bottom': '20px', 'font-size': '18px'})
        ]),
        html.Div(
            id='similar-images-container',
            style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'},
        ),
        html.H2(id='clicked-image-store', style={'text-align': 'center', 'margin-top': '20px'}),
        # html.Div(id='label-display')

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




# Callback to update uploaded image

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
        if not gui_utils.is_anomalous():
            underlined_label = "\u0332".join(label)
            label_mes = f'Your image label is: {underlined_label}'
        else:
            label_mes = f'Sorry, your image label is not in our training set'
        print(label)
    return label_mes, contents

@app.callback(
    [Output('uploaded-image', 'figure'),
     Output('label-after-crop', 'children')],
    Input('upload-data', 'contents'),
    Input('uploaded-image', 'relayoutData'),
    State('upload-data', 'filename'),
      # Add this State to get cursor data
    prevent_initial_call=True  # Add this to prevent the callback from running on initial load
)
def update_uploaded_image(contents,relayoutData, filename):
    global flag
    if contents is None:
        return fig, ''
    label_after_crop = ''
    # Process the uploaded image
    encoded_image = contents.split(',')[1]
    if relayoutData is not None and 'xaxis.range[0]' in relayoutData and flag==False:
        flag = True
        # Get cropping coordinates from cursor data
        x0 = relayoutData['xaxis.range[0]']
        x1 = relayoutData['xaxis.range[1]']
        y0 = relayoutData['yaxis.range[0]']
        y1 = relayoutData['yaxis.range[1]']
        # Crop the image based on the cursor data
        image_data = base64.b64decode(encoded_image)
        pillow_image = Image.open(io.BytesIO(image_data))
        cropped_image = pillow_image.crop((x0, y0, x1, y1))
        # Update the figure with the cropped image
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
                    source=cropped_image  # Use the cropped image
                )
            ]
        )
        encoded_image = fig['layout']['images'][0]['source']
        image_data = base64.b64decode(encoded_image.split(',')[1])
        pillow_image = Image.open(io.BytesIO(image_data))
        output_filename = 'output_image.png'
        cropped_image.save(output_filename, format='PNG')
        formated_image = gui_utils.format_image(encoded_image)
        label = gui_utils.predict_label(formated_image)
        underlined_label = "\u0332".join(label)
        label_after_crop = f'Your image label after crop is: {underlined_label}'
        print(label)
        return fig, label_after_crop
        print(6)
    else:
        flag = False
        print(6)
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
    return fig, label_after_crop


def create_image_div(similarity_phrase, img_vector):
    img_pil = gui_utils.decode_image(img_vector)  # Assuming you have a function to decode the image vector

    img_io = io.BytesIO()
    img_pil.save(img_io, format='PNG')
    img_data = img_io.getvalue()

    return html.Div([
        similarity_phrase,
        html.Img(
            src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
            style={'width': '150px', 'height': '150px', 'border': '2px solid black'}
        )
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'padding': '15px'})

# Callback to update similar images
@app.callback(
    [Output('similar-images-heading', 'style'),
     Output('similar-images-container', 'children')],
    [Input('confirm-no', 'n_clicks')]
)
def update_similar_images(label_mes):
    similar_images_container = None
    heading_style = {'display': 'none'}

    if label_mes:
        closest_images = gui_utils.similar_images_using_potential()
        similar_images_container = []

        ranking_phrases = ["Most Similar", "Second Most Similar", "Third Most Similar", "Fourth Most Similar"]

        for i, img_vector in enumerate(closest_images.values, start=0):
            similarity_phrase = ranking_phrases[i]
            similarity_button = html.Button(
                similarity_phrase,
                id={'type': 'ranking-button', 'index': i},
                style={'margin-bottom': '10px'}
            )
            image_div = html.Div([
                similarity_button,
                html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
                         style={'width': '150px', 'height': '150px', 'border': '2px solid black'})
            ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'padding': '15px'})
            similar_images_container.append(image_div)

        heading_style = {'text-align': 'right', 'margin-bottom': '20px'}

    return heading_style, similar_images_container


# to ask if we were right only when it predicted
# Update your toggle_confirmation_section callback
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

# Add a callback to reset n_clicks when the "Upload" button is clicked
@app.callback(
    Output('confirm-yes', 'n_clicks'),
    Output('confirm-no', 'n_clicks'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def reset_clicks_on_upload(contents):
    if contents is not None:
        return 0, 0
    return dash.no_update, dash.no_update


index_set = False
@app.callback(
    Output('clicked-image-store', component_property='children'),  # Store the clicked image index
    Input({'type': 'ranking-button', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def store_clicked_ranking_index(n_clicks_list):
    global index_set
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if any(n_clicks is not None for n_clicks in n_clicks_list):
        non_null_index = next(i for i in range(len(n_clicks_list)) if n_clicks_list[i] is not None)
        label = gui_utils.find_label_of_similar_image(non_null_index)
        underlined_label = "\u0332".join(label)
        print(label)
        index_set = True
        return f"Based on the image you selected, your label is {underlined_label}"
    else:
        return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)