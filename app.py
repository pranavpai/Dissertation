#!/usr/bin/env python
# coding: utf-8

# In[54]:


import dash_bootstrap_components as dbc
from dash import (Input, Output, State, html,  dcc, dash_table,
                  get_asset_url, ALL, MATCH)
from dash.exceptions import PreventUpdate
from jupyter_dash import JupyterDash as Dash
#==========================================

#==========================================
import time
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import io
import base64
import json
import numpy as np
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_canvas
from dash import html, dcc, dash_table


# In[55]:


mesh_size = 0.02


def prediction_plot(**kwargs):

    _data = kwargs['data']
    split_data = _data[0]
    model = kwargs['model']

    y_pred_train = (model[0].decision_function(split_data[0]) >
                    kwargs['threshold']).astype(int)
    y_pred_test = (model[0].decision_function(split_data[1]) >
                   kwargs['threshold']).astype(int)
    train_score = accuracy_score(y_true=split_data[2].astype(int),
                                 y_pred=y_pred_train)
    test_score = accuracy_score(y_true=split_data[3].astype(int),
                                y_pred=y_pred_test)

    scaled_threshold = kwargs['threshold'] * (model[1].max() -
                                              model[1].min()) + model[1].min()

    range = max(abs(scaled_threshold - model[1].min()),
                abs(scaled_threshold - model[1].max()))

    trace0 = go.Contour(x=np.arange(model[2].min(), model[2].max(), mesh_size),
                        y=np.arange(model[3].min(), model[3].max(), mesh_size),
                        z=model[1].reshape(model[2].shape),
                        zmin=scaled_threshold - range,
                        zmax=scaled_threshold + range,
                        hoverinfo='none',
                        showscale=False,
                        contours=dict(showlines=False),
                        colorscale='rdgy',
                        opacity=0.6)

    trace1 = go.Contour(x=np.arange(model[2].min(), model[2].max(), mesh_size),
                        y=np.arange(model[3].min(), model[3].max(), mesh_size),
                        z=model[1].reshape(model[2].shape),
                        showscale=False,
                        hoverinfo='none',
                        contours=dict(
                            showlines=False,
                            type='constraint',
                            operation='=',
                            value=scaled_threshold,
                        ),
                        name=f'Threshold ({scaled_threshold:.3f})',
                        line=dict(color='#454545'))

    trace2 = go.Scatter(x=split_data[0][:, 0],
                        y=split_data[0][:, 1],
                        mode='markers',
                        name=f'Training Data (accuracy={train_score:.3f})',
                        marker=dict(size=10,
                                    color=split_data[2].astype(int),
                                    colorscale='tealrose',
                                    line=dict(width=1)))

    trace3 = go.Scatter(x=split_data[1][:, 0],
                        y=split_data[1][:, 1],
                        mode='markers',
                        name=f'Test Data (accuracy={test_score:.3f})',
                        marker=dict(
                            size=10,
                            symbol='triangle-up',
                            color=split_data[3].astype(int),
                            colorscale='tealrose',
                            line=dict(width=1),
                        ))

    layout = go.Layout(
        xaxis=dict(
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        transition=dict(
            easing='exp-in-out',
            ordering="traces first",
            duration=500),
        yaxis=dict(
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        hovermode='closest',
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0))

    fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)

    return fig


def roc_curve_plot(**kwargs):

    _data = kwargs['data']
    split_data = _data[0]
    model = kwargs['model']

    y_score = model[0].decision_function(_data[1])
    fpr, tpr, thresholds = roc_curve(_data[2], y_score)

    auc_score = roc_auc_score(y_true=_data[2], y_score=y_score)

    fig = px.line(x=fpr, y=tpr)
    fig.update_layout(
        title={
            'text': f'ROC Curve (AUC = {auc_score:.3f})',
            'y': 0.5,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'
        },
        transition=dict(easing='cubic-in-out', duration=500),
        yaxis=dict(  #range=[0, 1],
            title='True Positive Rate',
            scaleanchor="x",
            scaleratio=1),
        xaxis=dict(  #range=[0, 1],
            title='False Positive Rate', constrain='domain'),
        hovermode='closest',
        height=400,
        showlegend=False,
        margin=dict(l=10, r=0, t=40, b=20))

    return fig


def confusion_matrix_plot(**kwargs):

    _data = kwargs['data']
    split_data = _data[0]
    model = kwargs['model']

    scaled_threshold = kwargs['threshold'] * (model[1].max() -
                                              model[1].min()) + model[1].min()
    y_pred_test = (model[0].decision_function(split_data[1]) >
                   scaled_threshold).astype(int).astype(str)

    matrix = confusion_matrix(y_true=split_data[3], y_pred=y_pred_test)
    mtx = matrix / matrix.sum()

    label_text = [["True Negative", "False Positive"],
                  ["False Negative", "True Positive"]]

    fig = px.imshow(mtx,
                    x=['X', 'y'],
                    y=['X', 'y'],
                    color_continuous_scale='sunsetdark',
                    zmin=0,
                    zmax=1,
                    aspect="auto")
    fig.update_traces(text=label_text, texttemplate="%{text}")

    fig.update_layout(xaxis_title="TRAIN",
                      yaxis_title="TEST",
                      hovermode='closest',
                      transition=dict(easing='sin-in-out', duration=500),
                      height=400,
                      margin=dict(l=10, r=20, t=40, b=20))

    return fig


# In[56]:


def parse_contents(contents, filename, header, usecols=None):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('csv'):
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),
                             usecols=usecols)
        elif filename.endswith('xls') or filename.endswith('xlsx'):
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), usecols=usecols)

    except:
        print('Somthing wrong with uploader.')

    if header:
        return df.columns

    else:
        return df


def handle_json(js):
    df = pd.DataFrame(json.loads(js)['objects'])
    df.fillna(value={'path': ''}, inplace=True)
    df['dot'] = df['path'].apply(lambda x: 1 if len(x) == 2 else 0)
    df['c'] = df['stroke'].apply(lambda x: 1 if x == '#509188' else 0)
    X = df[df['dot'] == 1]['pathOffset'].apply(pd.Series).to_numpy()
    y = df[df['dot'] == 1]['c'].to_numpy()
    X = (X - 250) / 500
    return X, y


# In[57]:


margin = 0.25
mesh_size = 0.02


def modeling(**kwargs):

    _data = kwargs['data']
    split_data = _data[0]

    

    x_min, x_max = _data[1][:, 0].min() - margin, _data[1][:, 0].max() + margin
    y_min, y_max = _data[1][:, 1].min() - margin, _data[1][:, 1].max() + margin

    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)

    xx, yy = np.meshgrid(xrange, yrange)

    clf = SVC(C=kwargs['cost'],
              kernel=kwargs['kernel'],
              degree=kwargs['degree'],
              gamma=kwargs['gamma'],)
#               shrinking=kwargs['shrinking']

    clf.fit(split_data[0], split_data[2])

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    return clf, Z, xx, yy, xrange, yrange


# In[58]:




test_size = 0.25


def sampling(**kwargs):
    if kwargs['dataset'] == 'moons':
        X, y = datasets.make_moons(n_samples=kwargs['sample_size'],
                                   noise=kwargs['noise'],
                                   random_state=5)

        return train_test_split(X,
                                y.astype(str),
                                test_size=kwargs['test_size'],
                                random_state=5), X, y

    elif kwargs['dataset'] == 'circles':
        X, y = datasets.make_circles(n_samples=kwargs['sample_size'],
                                     noise=kwargs['noise'],
                                     factor=0.5,
                                     random_state=1)
        return train_test_split(X,
                                y.astype(str),
                                test_size=kwargs['test_size'],
                                random_state=5), X, y

    elif kwargs['dataset'] == 'LS':
        X, y = datasets.make_classification(n_samples=kwargs['sample_size'],
                                            n_features=2,
                                            n_redundant=0,
                                            n_informative=2,
                                            random_state=2,
                                            n_clusters_per_class=1)

        rng = np.random.RandomState(2)
        X += kwargs['noise'] * rng.uniform(size=X.shape)

        return train_test_split(X,
                                y.astype(str),
                                test_size=kwargs['test_size'],
                                random_state=5), X, y

    else:
        return ValueError('error!')


def df_split(**kwargs):
    _df = kwargs['df']

    return train_test_split(
        _df[['x', 'y']].to_numpy(),
        _df['c'].to_numpy().astype(str),
        test_size=kwargs['test_size'],
        random_state=5), _df[['x', 'y']].to_numpy(), _df['c'].to_numpy()


def data_split(**kwargs):

    return train_test_split(kwargs['X'],
                            kwargs['y'].astype(str),
                            test_size=kwargs['test_size'],
                            random_state=5), kwargs['X'], kwargs['y']


# In[59]:


dataset = html.Div([
    html.Strong('Dataset'),
    dbc.RadioItems(id={
        'type': 'dataset_parameter',
        'index': 'dataset'
    },
                   className="btn-group",
                   inputClassName="btn-check",
                   labelClassName="btn btn-outline-primary",
                   labelCheckedClassName="active",
                   options=[
                       {
                           "label": "Moons",
                           "value": 'moons'
                       },
                       {
                           "label": "Linearly Separable",
                           "value": 'LS'
                       },
                       {
                           "label": "Circles",
                           "value": 'circles'
                       },
                   ],
                   value='LS')
])

sample_size = html.Div([
    html.Strong('Sample Size'),
    html.Br(),
    daq.Slider(
        id={
            'type': 'dataset_parameter',
            'index': 'sample_size'
        },
        min=100,
        max=500,
        value=100,
        step=100,
        marks={i * 100: i * 100
               for i in range(6)},
    )
])

noise_level = html.Div([
    html.Strong('Noise Level'),
    html.Br(),
    daq.Slider(id={
        'type': 'dataset_parameter',
        'index': 'noise'
    },
               max=1,
               value=0.4,
               step=0.1,
               marks={i / 5: i / 5
                      for i in range(1, 5)})
],
                       style={'margin-bottom': '15px'})

test_size = html.Div([
    html.Strong('Test Size'),
    html.Br(),
    daq.Slider(id={
        'type': 'dataset_parameter',
        'index': 'test_size'
    },
               max=0.5,
               value=0.25,
               step=0.05,
               marks={i / 10: i / 10
                      for i in range(1, 5)})
],
                     style={'margin-bottom': '15px'})

#==========================================


def reuse_table(i):
    return dash_table.DataTable(
        id={
            'type': 'tabs-table',
            'id': i
        },
        columns=[{
            "name":
            i,
            "id":
            i,
            "type":
            'numeric' if i != 'c' else 'text',
            'format':
            dash_table.Format.Format(precision=4,
                                     scheme=dash_table.Format.Scheme.fixed)
            if i not in ['s', 'c'] else None
        } for i in ['s', 'x', 'y', 'c']],
        #export_format ='csv',
        fixed_rows={'headers': True},
        style_table={
            'height': '300px',
            'overflow': 'auto'
        },
        page_action='none',
        style_cell={
            'width': '{}%'.format(100 / 4),
            'textOverflow': 'ellipsis',
            'overflow': 'hidden'
        })


#==========================================
#==========================================
#==========================================

tab_1_content = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row(dataset),
            html.Br(),
            dbc.Row([
                dbc.Col(sample_size, width=4),
                dbc.Col(noise_level, width=4),
                dbc.Col(test_size, width=4)
            ], )
        ]),
        className="mt-3",
    ),
    dbc.Card(
        dbc.CardBody([dcc.Loading(reuse_table('t1'))]),
        className="mt-3",
    )
])

tab_2_content = html.Div([
    dbc.Card(
        dbc.CardBody([
            dcc.Upload(
                id={
                    'type': 'uploader_parameter',
                    'index': 'uploader'
                },
                children=html.Div(
                    ['Drag and Drop or ',
                     html.A('Select Files')]),
                style={
                    #'width': '72%',
                    'height': '45px',
                    'lineHeight': '45px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                })
        ]),
        className="mt-3",
    ),
    dbc.Card(dcc.Loading([
        dbc.CardBody(
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Strong('x'),
                        html.Br(),
                        dcc.Dropdown(id={
                            'type': 'uploader_parameter',
                            'index': 'x'
                        })
                    ],
                             style={'margin-bottom': '15px'})),
                dbc.Col(
                    html.Div([
                        html.Strong('y'),
                        html.Br(),
                        dcc.Dropdown(id={
                            'type': 'uploader_parameter',
                            'index': 'y'
                        })
                    ],
                             style={'margin-bottom': '15px'})),
                dbc.Col(
                    html.Div([
                        html.Strong('c'),
                        html.Br(),
                        dcc.Dropdown(id={
                            'type': 'uploader_parameter',
                            'index': 'c'
                        })
                    ],
                             style={'margin-bottom': '15px'})),
                dbc.Col(
                    html.Div([
                        html.Strong('Test Size'),
                        html.Br(),
                        daq.Slider(id={
                            'type': 'uploader_parameter',
                            'index': 'test_size'
                        },
                                   max=0.5,
                                   value=0.25,
                                   step=0.05,
                                   marks={i / 10: i / 10
                                          for i in range(1, 5)})
                    ],
                             style={'margin-bottom': '15px'}))
            ]))
    ]),
             className="mt-3"),
    dbc.Card(dbc.CardBody([dcc.Loading(reuse_table('t2'))]), className="mt-3")
])

tab_3_content = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    [
                        html.Div(
                            [
                                dash_canvas.DashCanvas(
                                    id={
                                        'type': 'canvas_parameter',
                                        'index': 'canvas'
                                    },
                                    filename=
                                    '/assets/canvas_bg.png',  #get_asset_url('bg.png'),
                                    lineWidth=5,
                                    goButtonTitle='Generate',
                                    lineColor='#509188',
                                    #width=canvas_width,
                                    hide_buttons=[
                                        "zoom", "pan", "line", "pencil",
                                        "rectangle", "select"
                                    ])
                            ],
                            className='canvas_container')
                    ],
                    width=8),
                dbc.Col([
                    dbc.Row([
                        html.Div(dbc.Button('Toggle',
                                            id={
                                                'type': 'canvas_parameter',
                                                'index': 'toggle'
                                            }),
                                 style={'margin-top': '45px'})
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div([
                            html.Strong('Test Size'),
                            html.Br(),
                            daq.Slider(id={
                                'type': 'canvas_parameter',
                                'index': 'test_size'
                            },
                                       max=0.5,
                                       value=0.25,
                                       step=0.05,
                                       marks={
                                           i / 10: i / 10
                                           for i in range(1, 5)
                                       })
                        ])
                    ])
                ],
                        width=4)
            ])
        ]),
        className="mt-3",
    ),
    dbc.Card(dbc.CardBody([dcc.Loading(reuse_table('t3'))]), className="mt-3")
])

#==========================================
uploader_btn = html.Div([
    offcanvas_btn := dbc.Button("SELECT DATA",
                                outline=True,
                                color="primary",
                                size='lg'), offcanvas :=
    dbc.Offcanvas([
        tabs := dbc.Tabs(
            [
                tab_1 := dbc.Tab(label="Scikit-learn Datasets"), tab_2 :=
                dbc.Tab(label="Upload Data"), tab_3 :=
                dbc.Tab(label="Hand Drawn Datapoints")
            ],
            active_tab="tab-0",
        ), offcanvas_content := html.Div(),
        dbc.Card(
            html.Div([
                dbc.CardBody([save_btn := dbc.Button("SAVE", color="success")])
            ],
                     className="d-grid gap-2 mx-auto"),
            className="mt-3",
        )
    ],
                  placement='end',
                  is_open=False,
                  title='DATA UPLOAD',
                  style={'width': '85%'})
])

#==========================================
threshold = html.Div([
    html.Strong('Threshold'),
    html.Br(),
    daq.Knob(id={
        'type': 'svm_parameter',
        'index': 'threshold'
    },
             min=0,
             max=1,
             value=0.5,
             size=100), threshold_btn := dbc.Button("RESET THRESHOLD")
],
                     style={'margin-bottom': '15px'})

#==========================================
kernel = html.Div([
    html.Strong('Kernel'),
    html.Br(),
    dcc.Dropdown(id={
        'type': 'svm_parameter',
        'index': 'kernel'
    },
                 options={
                     'rbf': 'Radial basis function (RBF)',
                     'linear': 'Linear',
                     'poly': 'Polynomial',
                     'sigmoid': 'Sigmoid'
                 },
                 value='rbf',
                 style={'width': '75%'})
],
                  style={'margin-bottom': '15px'})

cost = html.Div(
    [
        html.Strong('Cost (C)'),
        html.Br(),
        daq.Slider(id={
            'type': 'svm_parameter',
            'index': 'cost_power'
        },
                   min=-2,
                   max=4,
                   value=0,
                   marks={i: 10**i
                          for i in range(-2, 5)}),
        html.Br(),
        daq.Slider(
            id={
                'type': 'svm_parameter',
                'index': 'cost_coef'
            },
            min=1,
            max=9,
            value=1,
            step=1,
            handleLabel={
                #"showCurrentValue": True,
                "label": "COST"
            })
    ],
    style={'margin-bottom': '15px'})

degree = html.Div([
    html.Strong('Degree'),
    html.Br(),
    daq.Slider(id={
        'type': 'svm_parameter',
        'index': 'degree'
    },
               min=2,
               max=10,
               value=2,
               step=1,
               marks={i: i
                      for i in range(2, 9, 2)})
],
                  style={'margin-bottom': '15px'})

gamma = html.Div(
    [
        html.Strong('Gamma'),
        html.Br(),
        daq.Slider(
            id={
                'type': 'svm_parameter',
                'index': 'gamma_power'
            },
            min=-5,
            max=0,
            value=-1,
            marks={i: 10**i
                   for i in range(-5, 1)},
        ),
        html.Br(),
        daq.Slider(
            id={
                'type': 'svm_parameter',
                'index': 'gamma_coef'
            },
            min=1,
            max=9,
            value=5,
            step=1,
            handleLabel={
                #"showCurrentValue": True,
                "label": "GAMMA",
                "style": {
                    "height": "15px"
                }
            })
    ],
    style={'margin-bottom': '15px'})

# shrinking = html.Div([
#     html.Strong('Shrinking'),
#     dbc.RadioItems(id={
#         'type': 'svm_parameter',
#         'index': 'shrinking'
#     },
#                    className="btn-group",
#                    inputClassName="btn-check",
#                    labelClassName="btn btn-outline-primary",
#                    labelCheckedClassName="active",
#                    options=[
#                        {
#                            "label": "Disable",
#                            "value": False
#                        },
#                        {
#                            "label": "Enable",
#                            "value": True
#                        },
#                    ],
#                    value=False)
# ])


# In[60]:


# -*- utf-8 -*-



#==========================================

#==========================================


#==========================================

#==========================================

#==========================================

#==========================================

#==========================================

#==========================================
#==========================================

app = Dash(__name__,
           title='SVM',
           update_title='Eating...',
           external_stylesheets=[dbc.themes.FLATLY])

server = app.server

app.config.suppress_callback_exceptions = True
#==========================================
#==========================================

#=============layout=======================

app.layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(
                        html.Img(src=get_asset_url('logo.png'),
                                 height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Support Vector Machines",
                                        className="ms-2")),
                ],
                        align="center",
                        className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
        ])),
    dbc.Row([
        dbc.Col(
            dbc.Container([
                html.Br(),
                dbc.Row(dbc.Col(fig_0 := dcc.Graph(), )),
                html.Br(),
                dbc.Row([
                    dbc.Col(fig_1 := dcc.Graph(), width=6, align='center'),
                    dbc.Col(fig_2 := dcc.Graph(), width=6, align='center'),
                ]),
                html.Br(),
                dbc.Row(
                    alert := dbc.Alert(is_open=False,
                                       dismissable=True,
                                       duration=2000,
                                       style={'padding-left': '45px'}), )
            ], ),
            width=8,
        ),
        dbc.Col(
            dbc.Container([
                html.Br(), uploader_btn,
                html.Br(), threshold,
                html.Br(), kernel,
                html.Br(), cost,
                html.Br(), degree,
                html.Br(), gamma,
                html.Br(),
#                 html.Br(), shrinking,
                html.Br()
            ]),
            width=4,
        )
    ]),
    dbc.Row(params := html.Div(style={'display': 'none'}))
])

#==========================================


#================callbacks=================
@app.callback(
    [
        Output({
            'type': 'uploader_parameter',
            'index': 'x'
        }, 'options'),
        Output({
            'type': 'uploader_parameter',
            'index': 'y'
        }, 'options'),
        Output({
            'type': 'uploader_parameter',
            'index': 'c'
        }, 'options')
    ],
    [Input({
        'type': 'uploader_parameter',
        'index': 'uploader'
    }, 'contents')],
    [State({
        'type': 'uploader_parameter',
        'index': 'uploader'
    }, 'filename')])
def update_output(data, filename):
    if data is None:
        raise PreventUpdate
    else:
        header = parse_contents(data, filename, header=True)
        return 3 * [header]


@app.callback(Output({
    'type': 'tabs-table',
    'id': 't2'
}, 'data'), [Input({
    'type': 'uploader_parameter',
    'index': ALL
}, 'value')], [
    State({
        'type': 'uploader_parameter',
        'index': ALL
    }, 'id'),
    State({
        'type': 'uploader_parameter',
        'index': 'uploader'
    }, 'contents'),
    State({
        'type': 'uploader_parameter',
        'index': 'uploader'
    }, 'filename')
])
def update_output(xyc, idx, uploaded_data, filename):
    data_params = {j['index']: xyc[i] for i, j in enumerate(idx)}
    if not all([v for k, v in data_params.items() if k not in ['uploader']
                ]) or not uploaded_data:
        raise PreventUpdate
    else:

        df0 = parse_contents(uploaded_data,
                             filename,
                             header=False,
                             usecols=[
                                 v for k, v in data_params.items()
                                 if k not in ['uploader', 'test_size']
                             ])

        df0.rename(
            columns={v: k
                     for k, v in data_params.items() if k != 'test_size'},
            inplace=1)

        data_params.update({'df': df0})

        data = df_split(**data_params)

        split_data = data[0]

        df1 = pd.DataFrame(split_data[0], columns=['x', 'y'])
        df1['c'] = split_data[2]
        df1['s'] = 'TRAIN'
        df2 = pd.DataFrame(split_data[1], columns=['x', 'y'])
        df2['c'] = split_data[3]
        df2['s'] = 'TEST'
        df = pd.concat([df1, df2])

        return df.to_dict('records')


@app.callback(Output({
    'type': 'tabs-table',
    'id': 't3'
}, 'data'), [
    Input({
        'type': 'canvas_parameter',
        'index': 'test_size'
    }, 'value'),
    Input({
        'type': 'canvas_parameter',
        'index': 'canvas'
    }, 'json_data')
])
def canvas_output(test_size, canvas_data):
    if not canvas_data:
        raise PreventUpdate

    X, y = handle_json(canvas_data)

    params = {'test_size': test_size, 'X': X, 'y': y}

    data = data_split(**params)

    split_data = data[0]

    df1 = pd.DataFrame(split_data[0], columns=['x', 'y'])
    df1['c'] = split_data[2]
    df1['s'] = 'TRAIN'
    df2 = pd.DataFrame(split_data[1], columns=['x', 'y'])
    df2['c'] = split_data[3]
    df2['s'] = 'TEST'
    df = pd.concat([df1, df2])

    return df.to_dict('records')


@app.callback(Output({
    'type': 'tabs-table',
    'id': 't1'
}, 'data'), [Input({
    'type': 'dataset_parameter',
    'index': ALL
}, 'value')], [State({
    'type': 'dataset_parameter',
    'index': ALL
}, 'id')])
def generate_data(value, idx):

    data_params = {j['index']: value[i] for i, j in enumerate(idx)}

    data = sampling(**data_params)

    split_data = data[0]

    df1 = pd.DataFrame(split_data[0], columns=['x', 'y'])
    df1['c'] = split_data[2]
    df1['s'] = 'TRAIN'
    df2 = pd.DataFrame(split_data[1], columns=['x', 'y'])
    df2['c'] = split_data[3]
    df2['s'] = 'TEST'
    df = pd.concat([df1, df2])
    return df.to_dict('records')


@app.callback(
    Output({
        'type': 'canvas_parameter',
        'index': 'canvas'
    }, 'lineColor'),
    [Input({
        'type': 'canvas_parameter',
        'index': 'toggle'
    }, 'n_clicks')],
    [State({
        'type': 'canvas_parameter',
        'index': 'canvas'
    }, 'lineColor')])
def toggle(n, c):
    if not all([n, c]):
        PreventUpdate

    if c == '#509188':
        return '#ff7070'
    else:
        return '#509188'


@app.callback(Output(params, 'children'),
              [Input({
                  'type': 'svm_parameter',
                  'index': ALL
              }, 'value')],
              [State({
                  'type': 'svm_parameter',
                  'index': ALL
              }, 'id')])
def params_update(value, idx):
    df = pd.DataFrame({'index': [i['index'] for i in idx], 'value': value})
    return dash_table.DataTable(df.to_dict('records'), [{
        "name": i,
        "id": i
    } for i in df.columns])


@app.callback([
    Output(fig_0, 'figure'),
    Output(fig_1, 'figure'),
    Output(fig_2, 'figure'),
    Output(alert, 'children'),
    Output(alert, 'is_open')
], [
    Input(save_btn, 'n_clicks'),
    Input({
        'type': 'svm_parameter',
        'index': ALL
    }, 'value')
], [
    State({
        'type': 'svm_parameter',
        'index': ALL
    }, 'id'),
    State({
        'type': 'dataset_parameter',
        'index': ALL
    }, 'id'),
    State({
        'type': 'dataset_parameter',
        'index': ALL
    }, 'value'),
    State({
        'type': 'uploader_parameter',
        'index': ALL
    }, 'id'),
    State({
        'type': 'uploader_parameter',
        'index': ALL
    }, 'value'),
    State({
        'type': 'uploader_parameter',
        'index': ALL
    }, 'contents'),
    State({
        'type': 'uploader_parameter',
        'index': ALL
    }, 'filename'),
    State({
        'type': 'canvas_parameter',
        'index': ALL
    }, 'json_data'),
    State({
        'type': 'canvas_parameter',
        'index': ALL
    }, 'value'),
    State(tabs, 'active_tab')
])
def params_update(n_clicks, value, idx, data_1_idx, data_1_value, data_2_idx,
                  data_2_value, uploaded_data, filename, canvas_data,
                  canvas_params, at):
    t1 = time.perf_counter()

    if at == 'tab-0':
        data_1_params = {
            j['index']: data_1_value[i]
            for i, j in enumerate(data_1_idx)
        }

        data = sampling(**data_1_params)

    elif at == 'tab-1':
        data_2_params = {
            j['index']: data_2_value[i]
            for i, j in enumerate(data_2_idx)
        }

        df0 = parse_contents(list(filter(None, uploaded_data))[0],
                             list(filter(None, filename))[0],
                             header=False,
                             usecols=[
                                 v for k, v in data_2_params.items()
                                 if k not in ['uploader', 'test_size']
                             ])

        df0.rename(columns={
            v: k
            for k, v in data_2_params.items() if k != 'test_size'
        },
                   inplace=1)

        data_2_params.update({'df': df0})

        data = df_split(**data_2_params)

    elif at == 'tab-2':
        X, y = handle_json(list(filter(None, canvas_data))[0], )

        split_params = {
            'test_size': list(filter(None, canvas_params))[0],
            'X': X,
            'y': y
        }

        data = data_split(**split_params)

    if not data:
        raise PreventUpdate

    params = {j['index']: value[i] for i, j in enumerate(idx)}

    params.update({'data': data})
    params.update({'cost': 10**params['cost_power'] * params['cost_coef']})
    params.update({'gamma': 10**params['gamma_power'] * params['gamma_coef']})

    model = modeling(**params)
    params.update({'model': model})

    fig_0 = prediction_plot(**params)
    fig_1 = roc_curve_plot(**params)
    fig_2 = confusion_matrix_plot(**params)

    t2 = time.perf_counter()

    alert_info = 'Takes {:.3} seconds'.format(t2 - t1)

    return fig_0, fig_1, fig_2, alert_info, True


@app.callback(Output(offcanvas, 'is_open'),
              [Input(offcanvas_btn, 'n_clicks'),
               Input(save_btn, 'n_clicks')], [State(offcanvas, 'is_open')])
def open_offcanvas(n, sn, is_open):
    if n or sn:
        return not is_open
    return is_open


@app.callback(Output(save_btn, 'disabled'),
              [Input({
                  'type': 'tabs-table',
                  'id': ALL
              }, 'is_loading')])
def btn_disabled(tabs_table):
    if any(tabs_table):
        return True
    else:
        return False


@app.callback(Output(offcanvas_content, "children"),
              [Input(tabs, "active_tab")])
def switch_tab(at):
    if at == 'tab-0':
        return tab_1_content
    elif at == 'tab-1':
        return tab_2_content
    elif at == 'tab-2':
        return tab_3_content
    return html.P("Something wrong...")


@app.callback(Output({
    'type': 'svm_parameter',
    'index': 'degree'
}, 'disabled'), [Input({
    'type': 'svm_parameter',
    'index': 'kernel'
}, 'value')])
def disable_param_degree(kernel):
    return kernel != 'poly'


@app.callback([
    Output({
        'type': 'svm_parameter',
        'index': 'gamma_power'
    }, 'disabled'),
    Output({
        'type': 'svm_parameter',
        'index': 'gamma_coef'
    }, 'disabled')
], [Input({
    'type': 'svm_parameter',
    'index': 'kernel'
}, 'value')])
def disable_param_gamma(kernel):
    _ = kernel not in ['rbf', 'poly', 'sigmoid']
    return _, _


@app.callback(
    Output({
        'type': 'svm_parameter',
        'index': 'cost_coef'
    }, 'marks'),
    [Input({
        'type': 'svm_parameter',
        'index': 'cost_power'
    }, 'value')])
def update_slider_svm_parameter_C_coef(power):
    scale = 10**power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(Output({
    'type': 'svm_parameter',
    'index': 'gamma_coef'
}, 'marks'), Input({
    'type': 'svm_parameter',
    'index': 'gamma_power'
}, 'value'))
def scale_param_gamma(power):
    scale = 10**power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(Output({
    'type': 'svm_parameter',
    'index': 'threshold'
}, 'value'), [Input(threshold_btn, 'n_clicks')], [State(fig_0, 'figure')])
def reset_threshold(n_clicks, fig):
    if n_clicks:
        Z = np.array(fig['data'][0]['z'])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.5
    return value


#==========================================
#==========================================
#==========================================
#==========================================
#==========================================


# In[61]:


app.run_server()


# In[ ]:




