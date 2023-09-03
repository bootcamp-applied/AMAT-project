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
<<<<<<< HEAD
        'height': '100vh'
    },
    children=[
        html.H1("Welcome to our Image Classifier", style={'text-align': 'center', 'margin-bottom': '20px'}),
=======
        'height': '100vh',
        'background-color': 'gray',  # Set your desired background color here
        'overflow-y': 'auto'
},
    children=[
        html.H1("Welcome to our Image Classifier",
                style={'text-align': 'center', 'margin-bottom': '20px', 'display': 'block', 'font-size': '50px',
                       'color': 'white', 'text-shadow': '4px 4px 6px #000000'}),
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
        dcc.Upload(id='upload-data', children=html.Button('Upload Image'), style={'margin-bottom': '20px'}),

        html.Div(
            id='uploaded-image-container',
<<<<<<< HEAD
            style={'text-align': 'center', 'border': '2px solid black'},
            children=[
=======
            style={'text-align': 'center', 'border': '4px solid black', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 1.5)'},
        children=[
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
                dcc.Graph(id='uploaded-image', config={'doubleClick': 'reset'})
            ]
        ),

        html.H2(id='image-label', style={'text-align': 'center', 'margin-top': '20px'}),
        html.H2(id='label-after-crop', style={'text-align': 'center', 'margin-top': '20px'}),
        confirmation_section,
        html.Div(id='similar-images-heading', children=[
<<<<<<< HEAD
            html.H2("Give us another chance", style={'text-align': 'center', 'margin-top': '20px'}),
=======
            html.H2("Give us another chance", style={'text-align': 'center', 'margin-top': '-20px'}),
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
            html.H2("Explore similar images from our training set. Click the button above the image with the same label",
                    style={'text-align': 'right', 'margin-bottom': '20px', 'font-size': '18px'})
        ]),
        html.Div(
            id='similar-images-container',
            style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'},
        ),
        html.H2(id='clicked-image-store', style={'text-align': 'center', 'margin-top': '20px'}),
        # html.Div(id='label-display')
<<<<<<< HEAD
=======

>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
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


<<<<<<< HEAD
# @app.callback(
#     [Output(component_id='image-label', component_property='children'),
#      Output(component_id='uploaded-image', component_property='data')],
#     [Input(component_id='upload-data', component_property='contents')]
# )
# def classify_label(contents):
#     label_mes = ''
#     if contents is not None:
#         formated_image = gui_utils.format_image(contents)
#         label = gui_utils.predict_label(formated_image)
#         label_mes = f'Your image label is: {label}'
#     return label_mes, contents
=======
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae


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
<<<<<<< HEAD
        label = gui_utils.is_anomalysis(formated_image)
        label_mes = f'Your image label is: {label}'
        print(label)
    return label_mes, contents
=======
        label = gui_utils.predict_label(formated_image)
        if not gui_utils.is_anomalous():
            underlined_label = "\u0332".join(label)
            label_mes = f'Your image label is: {underlined_label}'
        else:
            label_mes = f'Sorry, your image label is not in our training set'
        print(label)
    return label_mes, contents

>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
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
<<<<<<< HEAD
        pillow_image.save(output_filename, format='PNG')
        formated_image = gui_utils.format_image(encoded_image)
        label = gui_utils.predict_label(formated_image)
        label_after_crop = f'Your image label after crop is: {label}'
=======
        cropped_image.save(output_filename, format='PNG')
        formated_image = gui_utils.format_image(encoded_image)
        label = gui_utils.predict_label(formated_image)
        underlined_label = "\u0332".join(label)
        label_after_crop = f'Your image label after crop is: {underlined_label}'
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
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
<<<<<<< HEAD
# @app.callback(
#     Output('dummy-output', 'children'),
#
#     State('uploaded-image', 'figure')
# )
# def display_image_with_plt(uploaded_image_figure):
#
#         #if uploaded_image_figure and 'data' in uploaded_image_figure and 'source' in uploaded_image_figure['data'][0]:
#             encoded_image = uploaded_image_figure['layout']['images'][0]['source']
#             image_data = base64.b64decode(encoded_image.split(',')[1])
#             pillow_image = Image.open(io.BytesIO(image_data))
#             output_filename = 'output_image.png'
#             pillow_image.save(output_filename, format='PNG')
#             # You need to decode and display the image using plt.show() here
#             decoded_image = gui_utils.format_image(encoded_image)
#             image = np.reshape(decoded_image, (32, 32, 3))
#             plt.imshow(image)
#             plt.show()
#
=======
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae


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

<<<<<<< HEAD
#
# def update_similar_images(label_mes):
#     similar_images_container = None
#     heading_style = {'display': 'none'}
#
#     if label_mes:
#         closest_images = gui_utils.similar_images_using_potential()
#         similar_images_container = []
#
#         ranking_phrases = ["Most Similar", "Second Most Similar", "Third Most Similar", "Fourth Most Similar"]
#
#         for i, img_vector in enumerate(closest_images.values, start=0):
#             if i >= len(ranking_phrases):
#                 break
#
#             similarity_phrase = html.Div(ranking_phrases[i], style={'text-align': 'center', 'font-weight': 'bold'})
#             image_div = html.Div([
#                 similarity_phrase,
#                 html.Img(src=f"data:image/png;base64,{gui_utils.encode_image(img_vector)}",
#                          style={'width': '150px', 'height': '150px', 'border': '2px solid black'})
#             ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'padding': '15px'})
#             similar_images_container.append(image_div)
#
#         heading_style = {'text-align': 'right', 'margin-bottom': '20px'}
#
#     return heading_style, similar_images_container

# to ask if we were right only when it predicted
=======

# to ask if we were right only when it predicted
# Update your toggle_confirmation_section callback
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
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

<<<<<<< HEAD
# to plot the similar images if "no" is pressed
# @app.callback(
#     [Output('clicked-image-store', 'data'),
#      Output('image-label', 'children')],  # Store the clicked image index
#     Input({'type': 'image-click', 'index': ALL}, 'n_clicks'),
#
# )
# def store_clicked_image_index(n_clicks_list):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return None
#
#     triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if any(n_clicks is not None for n_clicks in n_clicks_list):
#         non_null_value = next(item for item in n_clicks_list if item is not None)
#         print(non_null_value)
#         return non_null_value, "cat"
#     else:
#         return dash.no_update, ""
# @app.callback(
#     Output('clicked-image-store', 'data'),  # Store the clicked image index
#     Input({'type': 'image-click', 'index': ALL}, 'n_clicks'),
#     State('clicked-image-store', 'data')
# )
# def store_clicked_image_index(n_clicks_list, prev_clicked_index):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return None
#
#     triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if any(n_clicks is not None for n_clicks in n_clicks_list):
#         non_null_value = next(i for i in range(len(n_clicks_list)) if n_clicks_list[i] is not None)
#         print(non_null_value)
#         label = gui_utils.find_label_of_similar_image(non_null_value)
#         return non_null_value
#     else:
#         return dash.no_update
#         ctx = dash.callback_context
#     if not ctx.triggered:
#         return None
#     if any(n_clicks is not None for n_clicks in n_clicks_list):
#         print(n_clicks_list)
#
#     # index = [ind for ind in n_clicks_list if ind == 1]
#     clicked_image_index = ctx.triggered[0]['prop_id'].split('.')[0]['index']
#     print(gui_utils.find_label_of_similar_image(clicked_image_index))
#     print(clicked_image_index)
#     return clicked_image_index



# @app.callback(
#     Output('clicked-image-store', 'data'),  # Store the clicked image index
#     Output('label-display', 'children'),  # Display the label to the user
#     Input({'type': 'image-click', 'index': ALL}, 'n_clicks'),
#     State('clicked-image-store', 'data')
# )
# def store_clicked_image_index(n_clicks_list, prev_clicked_index):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return None, None
#
#     triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if any(n_clicks is not None for n_clicks in n_clicks_list):
#         non_null_value = next(i for i in range(len(n_clicks_list)) if n_clicks_list[i] is not None)
#         label = gui_utils.find_label_of_similar_image(non_null_value)
#         print(label)
#         return non_null_value, label
#     else:
#         return dash.no_update, None
=======
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


>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
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
<<<<<<< HEAD
        print(label)
        index_set = True
        return f"Based on the image you selected, your label is {label}"
=======
        underlined_label = "\u0332".join(label)
        print(label)
        index_set = True
        return f"Based on the image you selected, your label is {underlined_label}"
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
    else:
        return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)