# import dash
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# import base64
# import cv2
# import numpy as np
# import pandas as pd
# # Sample dataset: replace this with your actual dataset
# # Assume you have a list of images for each class
#
# app = dash.Dash(__name__)
# df = pd.read_csv(r'../../data/processed/cifar-10-100.csv')
# # Define the layout
# app.layout = html.Div([
#
# html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),
#
#     dcc.Dropdown(
#         id='class-selector',
#         options=[
#             {"label": "airplane", "value": 0},
#             {"label": "automobile", "value": 1},
#             {"label": "bird", "value": 2},
#             {"label": "cat", "value": 3},
#             {"label": "deer", "value": 4},
#             {"label": "dog", "value": 5},
#             {"label": "frog", "value": 6},
#             {"label": "horse", "value": 7},
#             {"label": "ship", "value": 8},
#             {"label": "truck", "value": 9},
#             {"label": "fish", "value": 10},
#             {"label": "people", "value": 11},
#             {"label": "flowers", "value": 12},
#             {"label": "trees", "value": 13},
#             {"label": "fruit and vegetables", "value": 14}
#         ],
#         value=0,
#         multi=False,
#         style={'width': '40%'}
#     ),
#     html.Div(id='image-grid', children=[])#,
#     # html.Br(),
#     # dcc.Textarea(id='my_img_map', figure={})
# ])
# def encode_image(image):
#     image = image.reshape(3, 32, 32).transpose(1, 2, 0)
#     _, image = cv2.imencode('.png', image)
#     return base64.b64encode(image).decode()
# @app.callback(
#     Output('image-grid', 'children'),
#     Input('class-selector', 'value')
# )
# def update_image_grid(selected_class):
#     #can delete
#     print(selected_class)
#     print(type(selected_class))
#     ###
#     dff = df.copy()
#     dff = dff[dff["label"] == selected_class]
#     dff = dff.iloc[:9, 2:]
#     images = dff
#     image_components = []
#     for i, image in enumerate(images):
#         image_base64 = encode_image(image)
#         image_components.append(
#             html.Img(id='readed-image', src='data:image/png;base64,{}'.format(image_base64),alt=image_base64, style={'width': '150px', 'height': '150px'})
#         )
#     # Create a grid layout with 3 rows and 3 columns
#     grid_layout = html.Div(
#         [
#             html.Div(image_components[0:3], style={'display': 'flex'}),
#             html.Div(image_components[3:6], style={'display': 'flex'}),
#             html.Div(image_components[6:9], style={'display': 'flex'}),
#         ],
#         style={'display': 'flex', 'flexDirection': 'column'}
#     )
#     return grid_layout
# if __name__ == '__main__':
#     app.run_server(debug=True)


