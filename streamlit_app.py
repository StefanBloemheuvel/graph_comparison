import streamlit as st
st.set_page_config(layout="centered")
import pandas as pd
import numpy as np
import networkx as nx
from geoconnector.newest_graph_maker import graph_generator
import os 
import streamlit.components.v1 as components
import time
from io import BytesIO
from datetime import datetime

st.set_option('deprecation.showPyplotGlobalUse', False)

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1000, max_width_100_percent: bool = False,
        padding_top: int = 10, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )
        



set_page_container_style(
        max_width = 1000, max_width_100_percent = True,
        padding_top = 50, padding_right = 10, padding_left = 5, padding_bottom = 10
)


st.sidebar.title('Graph Construction')

algorithm = st.sidebar.radio(
    "Which algorithm would you like to use?",
    ("Gabriel", "knn", "Relative Neighborhood","Gaussian","Optics","MinMax","Kmeans")
)
dataset = st.sidebar.radio(
    "Which dataset would you like to use?",
    ("la", "bay", "air","ci","cw"), horizontal=True
)

if algorithm not in ['Gabriel','Relative Neighborhood']:
    st.sidebar.write(f'This algorithm needs the parameters:')
    
    if algorithm in ['MinMax','Gaussian','Distance-Delimiter','greedy-spanner']:
        parameter = st.sidebar.slider('Select a value for k', 0.05, 1.0, 0.4)
    else:
        parameter = st.sidebar.slider('Select a value for k', 1, 60, 2)

# node_size = st.sidebar.slider('Select node size',1,30,10)
# edge_width = st.sidebar.slider('Select edge width',1,10,1)

selected_file = [f'sensor_locations/sensor_locations_{dataset}.csv']
# @st.cache_data
graph_generator_obj = graph_generator()

start = time.time()

with st.spinner('Wait for it...'):
    if algorithm == 'Gabriel':
        graph_generator_obj.gabriel(selected_file[0])
    if algorithm == 'Relative Neighborhood':
        graph_generator_obj.relative_neighborhood(selected_file[0])
    if algorithm == 'knn':
        graph_generator_obj.knn_unweighted(selected_file[0], k=parameter)
    # if algorithm == 'Distance-Delimiter':
        # graph_generator_obj.distance_limiter(selected_file[0], k=parameter)
    if algorithm == 'Optics':
        graph_generator_obj.optics(selected_file[0], min_samples=parameter)
    # if algorithm == 'DBSCAN':
        # graph_generator_obj.dbscan(selected_file[0], min_samples=parameter)
    if algorithm == 'MinMax':
        graph_generator_obj.minmax(selected_file[0], cutoff=parameter)
    if algorithm == 'Gaussian':
        graph_generator_obj.gaussian(selected_file[0], normalized_k=parameter)
    if algorithm == 'Kmeans':
        graph_generator_obj.kmeans(selected_file[0], num_clusters=parameter)
    # if algorithm == 'greedy-spanner':
        # graph_generator_obj.greedy_spanner(selected_file[0], t=parameter)
    
end = time.time()


st.sidebar.write(f'# RESULTS')
st.sidebar.write(f'Number of nodes = {nx.number_of_nodes(graph_generator_obj.networkx_graph)}')
st.sidebar.write(f'Number of edges = {nx.number_of_edges(graph_generator_obj.networkx_graph)}')
st.sidebar.write(f'Density = {nx.density(graph_generator_obj.networkx_graph):.2f}')
st.sidebar.write(f'Computation Time = {end - start:.2f} sec.')

st.sidebar.write(f'Last run = {datetime.now().strftime("%H:%M:%S")}')

# st.pyplot(graph_generator_obj.plot_graph(node_size=1, edge_width=1))

input_resize = 0.9
resize = input_resize * 0.999
col1, _ = st.columns((resize / (1 - resize), 1), gap="small")
# col1.pyplot(graph_generator_obj.plot_graph(node_size=1, edge_width=1))
col1.pyplot(graph_generator_obj.plot_graph(node_size=1, edge_width=1))


