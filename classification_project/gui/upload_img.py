import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash_image import Dash, dcc, html, Input, Output

from classification_project.study.use_Visualization import plot_images_to_given_label

app = Dash(__name__)
df = pd.read_csv(r'../../data/processed/cifar_10_100.csv')

#df = df.groupby(['label', 'is_train'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
print(df[:5])
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_class",
                 options=[
                     {"label": "airplane", "value": 0},
                     {"label": "automobile", "value": 1},
                     {"label": "bird", "value": 2},
                     {"label": "cat", "value": 3},
                     {"label": "deer", "value": 4},
                     {"label": "dog", "value": 5},
                     {"label": "frog", "value": 6},
                     {"label": "horse", "value": 7},
                     {"label": "ship", "value": 8},
                     {"label": "truck", "value": 9},
                     {"label": "fish", "value": 10},
                     {"label": "people", "value": 11},
                     {"label": "flowers", "value": 12},
                     {"label": "trees", "value": 13},
                     {"label": "fruit and vegetables", "value": 14}],
                 multi=False,
                 value=0,
                 style={'width': "50%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),
    dcc.Textarea(id='my_img_map', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_img_map', component_property='figure')],
    [Input(component_id='slct_class', component_property='value')]
)
def update_view(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The class chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["label"] == option_slctd]
    dff = dff[:9, 2:]

    # Plotly Express

    fig = px.imshow(
        img = dff,
        labels =dict(x='images', color= 'red'),
        data_frame = dff,
        facet_col_wrap = 3,
        facet_col_spacing = 0.2,
        facet_row_spacing = 0.2,
        title = "image from class",
        width = 2000,
        height = 2000,
        aspect = 'equal',
        contrast_rescaling ='infer',
        binary_format='png'
    )#→ plotly.graph_objects._figure.Figure


    return container, fig


# app.run_server()
