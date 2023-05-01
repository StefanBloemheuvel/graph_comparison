#%%
import os
import sys
import pandas as pd
from geopy.distance import geodesic
import networkx as nx
import numpy as np
np.set_printoptions(suppress=True)
import warnings
from libpysal import weights
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap as Basemap
from math import radians, sin, cos, sqrt, asin
from scipy.spatial.distance import  pdist, squareform
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import OPTICS
import scipy as sp
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from sklearn.preprocessing import minmax_scale
from dtaidistance import dtw
import time
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import entropy
from minepy import pstats, cstats
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
# other dependencies
# geopandas

class graph_generator:
    def load_location_data(self, path):
        self.path = path
        
        if self.path.split('.')[-1] == 'csv':
            # self.data = pd.read_csv(path, dtype={'station': int, 'lat':float, 'lon':float})
            self.data = pd.read_csv(path, dtype={'station': 'str', 'lat': 'float', 'lon':'float'})
        # if self.path.split('.')[-1] == 'npy':
        #     # self.data = pd.read_csv(path, dtype={'station': int, 'lat':float, 'lon':float})
        #     self.data = np.load(path)
        else:
            print('incorrect path!')
        
    def minmax(self, path, cutoff=0.3):
        print(f'Package Print: went for minmax with cutoff = {cutoff}')
        self.created_with = 'minmax'
         
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
            
        for idx1, itm1 in self.data[['station','lat','lon']].iterrows():
            for idx2, itm2 in self.data[['station','lat','lon']].iterrows():
                        pos1 = (itm1[1],itm1[2])
                        pos2 = (itm2[1],itm2[2])
                        distance = geodesic(pos1, pos2,).km
                        if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                            graph.add_edge(itm1[0], itm2[0], weight=distance)


        min_weight, max_weight = min(nx.get_edge_attributes(graph, "weight").values()), max(nx.get_edge_attributes(graph, "weight").values())

        for i,j in enumerate(graph.edges(data=True)):
            graph[j[0]][j[1]]['weight'] = 0.98 - (graph[j[0]][j[1]]['weight'] - min_weight) / (max_weight - min_weight)

        graph.remove_edges_from((e for e, w in nx.get_edge_attributes(graph,'weight').items() if w < cutoff))

        
        self.networkx_graph = graph
    
    def polygon_area(self, radius = 6378137):
        """
        Computes area of spherical polygon, assuming spherical Earth. 
        Returns result in ratio of the sphere's area if the radius is specified.
        Otherwise, in the units of provided radius.
        lats and lons are in degrees.
        """
        
        lats = np.deg2rad(self.data['lat'].values)
        lons = np.deg2rad(self.data['lon'].values)

        # Line integral based on Green's Theorem, assumes spherical Earth

        #close polygon
        if lats[0]!=lats[-1]:
            lats = append(lats, lats[0])
            lons = append(lons, lons[0])

        #colatitudes relative to (0,0)
        a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
        colat = 2*arctan2( sqrt(a), sqrt(1-a) )

        #azimuths relative to (0,0)
        az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

        # Calculate diffs
        # daz = diff(az) % (2*pi)
        daz = diff(az)
        daz = (daz + pi) % (2 * pi) - pi

        deltas=diff(colat)/2
        colat=colat[0:-1]+deltas

        # Perform integral
        integrands = (1-cos(colat)) * daz

        # Integrate 
        area = abs(sum(integrands))/(4*pi)

        area = min(area,1-area)
        if radius is not None: #return in units of radius
            print('came here')
            return round(area * 4*pi*radius**2 / 1e+6, 2)
        else: #return in ratio of sphere total area
            print('came here 2')
            return round(area,2)
    
    def gaussian(self, path, normalized_k):
        
        print('Package Print: went for gaussian')
        self.created_with = 'gaussian'
        
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()
        for k in self.data[['station','lat','lon']].itertuples():
            graph.add_node(k[1], pos=(k[2],k[3])) 
        
        for idx1, itm1 in self.data[['station','lat','lon']].iterrows():
            for idx2, itm2 in self.data[['station','lat','lon']].iterrows():
                    pos1 = (itm1[1],itm1[2])
                    pos2 = (itm2[1],itm2[2])
                    distance = geodesic(pos1, pos2,).km
                    if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                        graph.add_edge(itm1[0], itm2[0], weight=distance)
                        
        distance_df =  nx.to_pandas_edgelist(graph)
        distance_df['weight'] = minmax_scale(distance_df['weight'])
        print(distance_df.head(5))
        print('did this')
        sensor_ids = self.data['station'].values
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf

        # Builds sensor id to index map.
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        # Fills cells in the matrix with distances.
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

        # Calculates the standard deviation as theta.
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0
        self.networkx_graph = nx.from_numpy_matrix(adj_mx)
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        

        
        
    def knn_unweighted(self, path, k=4):   
        print('Package Print: went for knn_unweighted')
        self.created_with = 'knn_unweighted'
         
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()
        for i in self.data.iterrows():
            graph.add_node(i[0], pos=(i[1][1],i[1][2]))
            
        self.data['lat_rad'] = np.deg2rad(self.data['lat'])
        self.data['lon_rad'] = np.deg2rad(self.data['lon'])
        
        tree = BallTree(self.data[['lat_rad','lon_rad']], metric="haversine")
        distances, indices = tree.query(self.data[['lat_rad','lon_rad']], k=k)
        distances[:,1:] = distances[:,1:] * 6371
        
        # Method 1
        for i in indices:
            [graph.add_edge(i[0],j, weight=1) for j in i[1:]]
            
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
        self.tree = tree
    
    def NormalizeData(self, input_data):
        return (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    
    def knn_weighted(self, path, k=5):
        print('Package Print: went for knn_weighted')
        self.created_with = 'knn_weighted'
        
        # Load the data
        self.load_location_data(path)
        old_names = self.data['station'].copy()
        self.data['station'], tesst_key = pd.factorize(self.data['station'])

        graph = nx.Graph()
        for i in self.data[['station','lat','lon']].itertuples():
            graph.add_node(i[1], pos=(i[2],i[3]))
        
        pos = nx.get_node_attributes(graph,'pos')
        names = [i for i in graph.nodes()]
        indexes = [i for i in range(0,nx.number_of_nodes(graph))]
        a_dictionary = dict(zip(indexes, names))

        self.data['lat_rad'] = np.deg2rad(self.data['lat'])
        self.data['lon_rad'] = np.deg2rad(self.data['lon'])
        
        tree = BallTree(self.data[['lat_rad','lon_rad']], metric="haversine")
        distances, indices = tree.query(self.data[['lat_rad','lon_rad']], k=k)
        distances[:,1:] = distances[:,1:] * 6371
        distances = self.NormalizeData(distances)

        y = indices.copy()
        for item, v in a_dictionary.items():
            indices[y == item] = v
        
        depth = k
        for indices_i,distances_j in zip(indices,distances):
            for k in range(1,depth):
                graph.add_edge(indices_i[0],indices_i[k],weight=distances_j[k])

        graph = nx.relabel_nodes(graph, dict(zip(range(0,len(old_names)),old_names)))

        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
        self.tree = tree
        
    def gabriel(self, path):
        print('Package Print: went for gabriel')
        # Load the data
        self.created_with = 'gabriel'

        self.load_location_data(path)
        
        tri = Delaunay(self.data[['lat','lon']].values)
        node_names = self.data['station']
        self.simplices = tri.simplices
        
        edges = []
        for i in tri.simplices:
            for j in range(0,3):
                for k in range(0,3):
                    if j != k:
                        edges.append((i[j],i[k]))
        new_df = pd.DataFrame(edges).drop_duplicates().sort_values([0, 1]).groupby(0)[1].apply(list).to_dict()
    
        lil = lil_matrix((tri.npoints, tri.npoints))
        indices, indptr = tri.vertex_neighbor_vertices
        for k in range(tri.npoints):
            lil.rows[k] = indptr[indices[k]:indices[k+1]]
            lil.data[k] = np.ones_like(lil.rows[k])  # dummy data of same shape as row
        
        coo = sp.sparse.csr_matrix(lil.toarray()).tocoo()
        conns = np.vstack((coo.row, coo.col)).T
        
        delaunay_conns = np.sort(conns, axis=1)
        
        c = tri.points[delaunay_conns]
        m = (c[:, 0, :] + c[:, 1, :])/2
        r = np.sqrt(np.sum((c[:, 0, :] - c[:, 1, :])**2, axis=1))/2
        tree = sp.spatial.cKDTree(self.data[['lat','lon']].values)
        n = tree.query(x=m, k=1)[0]
        g = n >= r*(0.999)  # The factor is to avoid precision errors in the distances
        gabriel_conns = delaunay_conns[g]
        graph = nx.from_edgelist(gabriel_conns)
        
        graph = nx.relabel_nodes(graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        self.networkx_graph = graph
        
    def distance_limiter(self, path, k):
        print('Package Print: went for Distance Limiter')
        self.created_with = 'distance_limiter'
        self.load_location_data(path)
        
        self.distance_limiter_obj = weights.DistanceBand.from_array(self.data[['lat','lon']], threshold=k)
        self.networkx_graph = self.distance_limiter_obj.to_networkx()
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
    def relative_neighborhood(self, path):
        print('Package Print: went for relative neighborhood')
        self.created_with = 'relative neighborhood'
        
        # Load the data
        self.load_location_data(path)
        
        graph = weights.Relative_Neighborhood(self.data[['lat','lon']]).to_networkx()
        # print(nx.info(graph))
        
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))

    def load_sensor_data(self, sensor_data_path):
        if sensor_data_path.split('.')[-1] == 'npy':
            self.sensor_data = np.load(sensor_data_path)
        else:
            print(f'error, this data is not npy, it is {sensor_data_path.split(".")[-1]} ')
            
    
    def from_signal(self, location_path, data_path, variant, clips, threshold):
        print(f'Package Print: went for signal based graph with variant = {variant}')
        self.created_with = variant
        
        # Load the data
        self.load_location_data(location_path)
        self.load_sensor_data(data_path)
        
        print(f'data shape = {self.sensor_data.shape}')
        graph = nx.Graph()
        for i in self.data[['station','lat','lon']].itertuples():
            graph.add_node(i[1], pos=(i[2],i[3]))
        
        start = time.time()
        if clips == True:
            print(f'went for clips')
            self.sensor_data = self.sensor_data.astype('double')
            # self.sensor_data = self.sensor_data[0:12000,:,:]
            # self.sensor_data = self.sensor_data[0:int(self.sensor_data.shape[0]/20),:,:]
            # length = int(self.sensor_data.shape[0] / 1000)
            length = 24 #in case of bay and la it is 1 hour, in case of air dataset it is 12 hours.
            sens = self.sensor_data.shape[1]

            print(f'length shape = {length}')
            print(f'sens shape = {sens}')
            print(f'data shape = {self.sensor_data.shape}')
            
            
            idx_last = -(self.sensor_data.shape[0] % length)
            if idx_last < 0:    
                clips = self.sensor_data[:idx_last].reshape(-1, self.sensor_data.shape[1],length)
            else:
                clips = self.sensor_data[idx_last:].reshape(-1, self.sensor_data.shape[1],length)                
            print(f'shape of clips = {clips.shape}')
            
            np.random.seed(1)
            random.seed(1)
            clips = clips[np.random.choice(range(clips.shape[0]), 50, replace=False),:,:]
            
            print(f'shape of clips = {clips.shape}')
            collection_of_graphs = np.empty((clips.shape[0],sens,sens))
            print(f'shape of collection of graphs = {collection_of_graphs.shape}')
            for i,j in enumerate(clips):
                if i % 50 == 0:
                    print(f'iteration {i} of {variant}')
                try:
                    if variant == 'dtw':
                        R1 = dtw.distance_matrix_fast(j)
                    if variant == 'correlation':
                        R1 = np.corrcoef(j)
                    if variant == 'mic':
                        mic_p, tic_p =  pstats(j, alpha=9, c=5, est="mic_e")

                        R1 = np.zeros((self.sensor_data.shape[1],self.sensor_data.shape[1]))
                        ind = np.triu_indices(self.sensor_data.shape[1], k=1)
                        R1[ind] = mic_p
                        R1 = R1 + R1.T - np.diag(np.diag(R1))
                        
                    collection_of_graphs[i] = R1
                
                except Exception as e:
                    print(f'error was at {i,j} {mic_p.shape, R1.shape} = {e}')
                    continue             
            
        else:
            np.random.seed(1)
            random.seed(1)
            self.sensor_data = self.sensor_data[np.random.choice(range(self.sensor_data.shape[0]), 50, replace=False),:,:,:]
            print(f'sensor shape after random selection = {self.sensor_data.shape}')
            print(f'went for segmented version')
            collection_of_graphs = np.empty((self.sensor_data.shape[0],self.sensor_data.shape[1],self.sensor_data.shape[1]))
            print(f'{collection_of_graphs.shape}')
            print(f'started with the procedure')
            for i,j in enumerate(self.sensor_data):
                if i % 50 == 0:
                    print(f'iteration {i}')
                try:
                    if variant == 'dtw':
                        R1 = dtw.distance_matrix_fast(j[:,:1000,0])
                    if variant == 'correlation':
                        R1 = np.corrcoef(j[:,:1000,0])
                    if variant == 'mic': # alpha was 5
                        mic_p, tic_p =  pstats(j[:,:1000,0], alpha=9, c=5, est="mic_e")

                        R1 = np.zeros((self.sensor_data.shape[1],self.sensor_data.shape[1]))
                        ind = np.triu_indices(self.sensor_data.shape[1], k=1)
                        R1[ind] = mic_p
                        R1 = R1 + R1.T - np.diag(np.diag(R1))
                                
                    collection_of_graphs[i] = R1
                        
                except Exception as e:
                    print(e)
                    continue 
        
        end = time.time()
        print("Time consumed in working: ",int(end - start))
        print()
        
        collection_of_graphs = np.nan_to_num(collection_of_graphs)
        print(f'maximum before the procedure = {collection_of_graphs.flatten().max()}')
        collection_of_graphs = np.sort(collection_of_graphs, axis= 0) # this is from small to large!
        
        print(collection_of_graphs[0,0:5,0:5])
        
        if variant == 'correlation' or variant == 'mic':
            collection_of_graphs = collection_of_graphs[-20:,:,:]
        if variant == 'dtw':
            collection_of_graphs = collection_of_graphs[:,:,:]
        
        collection_of_graphs = collection_of_graphs.mean(axis=0)
        print(f'maximum after taking the mean = {collection_of_graphs.flatten().max()}')
        print(collection_of_graphs[0:5,0:5])
        
        if variant == 'dtw' or variant == 'mic':
            collection_of_graphs = 1 - (collection_of_graphs - collection_of_graphs.min()) / (collection_of_graphs.max() - collection_of_graphs.min())
        collection_of_graphs[collection_of_graphs < threshold] = 0
        
        print(f'{collection_of_graphs[0:5,0:5]}')
        self.networkx_graph = nx.from_numpy_array(collection_of_graphs)
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))

    
    def kmeans(self, path, num_clusters):
        print('Package Print: went for kmeans own')
        self.created_with = 'kmeans_own'
        
        # Load the data
        self.load_location_data(path)
        
        print('went for kmeans_own')
        graph = nx.Graph()

        centroids, mean_dist = kmeans(self.data[['lat','lon']], num_clusters, seed=1)
        clusters, dist = vq(self.data[['lat','lon']], centroids)
        self.data['cluster'] = clusters

        for k in self.data[['station','lat','lon','cluster']].itertuples():
            graph.add_node(k[1], pos=(k[2],k[3]), cluster=k[4])    
        
        for node_r in graph.nodes(data=True):
            for node in graph.nodes(data=True):
                if node != node_r and node[1]['cluster'] == node_r[1]['cluster'] and node_r[1]['cluster'] != -1 and node[1]['cluster'] != -1:
                    graph.add_edge(node[0], node_r[0], weight=1)
                
        self.networkx_graph = graph
        
        
  
    # epcs was 2
    def dbscan(self, path, visualize=False, eps=4, min_samples=2):
        print(f'Package Print: went for dbscan with {eps=} and {min_samples}')
        self.created_with = 'dbscan'
        
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
        number_of_nodes = self.data.shape[0]

        X=self.data.loc[:,['lat','lon']]
        distance_matrix = squareform(pdist(X, (lambda u,v: self.haversine(u,v))))

        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # using "precomputed" as recommended by @Anony-Mousse
        y_db = db.fit_predict(distance_matrix)

        X['cluster'] = y_db
        print(f"number of clusters = {X['cluster'].nunique()}")
        print(f'number of points in no cluster = { X[X["cluster"] == -1].shape[0]}')
        
        if visualize==True:
            plt.scatter(X['lon'], X['lat'], c=X['cluster'],cmap='viridis')
            plt.show()
        
        # graph = nx.Graph()
        adj = np.zeros((number_of_nodes,number_of_nodes))
        for i,j in enumerate(X['cluster']):
            for k,l in enumerate(X['cluster']):
                if j == l and j != -1 and l != -1:
                    graph.add_edge(i,k, weight=1)

        graph.remove_edges_from(nx.selfloop_edges(graph))
        # pos = nx.get_node_attributes(graph,'pos')
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
    def optics(self, path, min_samples=2):
        print('Package Print: went for optics')
        self.created_with = 'optics'

        visualize=False
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
        number_of_nodes = self.data.shape[0]

        X=self.data.loc[:,['lat','lon']]
        distance_matrix = squareform(pdist(X, (lambda u,v: self.haversine(u,v))))

        optics_clustering = OPTICS(min_samples=min_samples, metric='precomputed')
        y_db = optics_clustering.fit_predict(distance_matrix)

        self.data['cluster'] = y_db
        print(f"number of clusters = {self.data['cluster'].nunique()}")
        print(f'number of points in no cluster = { self.data[self.data["cluster"] == -1].shape[0]}')
        
        if visualize==True:
            plt.scatter(self.data['lon'], self.data['lat'], c=self.data['cluster'],cmap='viridis')
            plt.show()
        
        # graph = nx.Graph()
        adj = np.zeros((number_of_nodes,number_of_nodes))
        for i,j in enumerate(self.data['cluster']):
            for k,l in enumerate(self.data['cluster']):
                # if j == l:
                if j == l and j != -1 and l != -1:
                
                    graph.add_edge(i,k, weight=1)

        graph.remove_edges_from(nx.selfloop_edges(graph))
        pos = nx.get_node_attributes(graph,'pos')
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))



        
    def create_adjacency_matrix(self, fill_diagonal = False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            epsilon = 5e-8
            adj = np.asarray(nx.adjacency_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense())
            
            if fill_diagonal == True:
                print(f'filled diagonal with ones')
                np.fill_diagonal(adj, 1)
            # if normalize_axis == True:
                # adj = adj / (adj.sum(normalize_axis, keepdims=True) + epsilon)
                
            self.adjacency_matrix = adj

    def create_normalized_laplacian_matrix(self):
        print('created normalized laplacian')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.normalized_laplacian_matrix = nx.normalized_laplacian_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense()
            np.fill_diagonal(self.normalized_laplacian_matrix,1)

    def plot_graph(self, plot_labels=False, node_size=30, font_size=7, edge_width=5):
        
        lat_distance = abs(abs(np.max(self.data['lat']))- abs(np.min(self.data['lat'])) )
        lon_distance = abs(abs(np.max(self.data['lon']))- abs(np.min(self.data['lon'])) )
        
        print(f'{lat_distance=},{lat_distance=}')
        m = Basemap(
            projection='merc',
            llcrnrlon=np.min(self.data['lon'])-lon_distance/5,
            llcrnrlat=np.min(self.data['lat'])-lat_distance/5,
            urcrnrlon=np.max(self.data['lon'])+lon_distance/5,
            urcrnrlat=np.max(self.data['lat'])+lat_distance/5,
            resolution='i', #could also be l
            suppress_ticks=False)
        
        
        haversine_data = {'lon':  [np.min(self.data['lon']), np.max(self.data['lon'])],
            'lat': [np.min(self.data['lat']), np.max(self.data['lat'])],
            }
        haversine_data = pd.DataFrame(haversine_data)
        max_distance = squareform(pdist(haversine_data, (lambda u,v: self.haversine(u,v))))[0,1]
        print(f'diagonal distance = {max_distance:.2f}km')
        
        mx,my=m(self.data['lon'].values,self.data['lat'].values)
        pos = {}
        for count, elem in enumerate (self.data['station']):
            pos[elem] = (mx[count], my[count])
            
        fig = plt.figure()
        # m.etopo()
        # m.shadedrelief()
        m.drawcountries(color='#a9acb1')
        m.drawstates()
        m.drawrivers()
        m.drawcoastlines()
        m.drawmapboundary(fill_color='#2081C3')
        m.fillcontinents(color='#f0efdb',lake_color='#2081C3')
        m.drawparallels(range(int(np.min(self.data['lat'])), int(np.max(self.data['lon'])), 2),linewidth=1.0)
        m.drawmapscale((np.min(self.data['lon'])-lon_distance/10)+max_distance/1000+1.69,np.min(self.data['lat'])-lat_distance/10,(np.min(self.data['lon'])-lon_distance/5),(np.min(self.data['lat'])-lat_distance/5),100,barstyle='fancy') #max_distance/5
        nx.draw_networkx_nodes(G = self.networkx_graph, pos = pos, 
                        node_color = 'r', alpha = 1, node_size = node_size)
        if plot_labels:
            nx.draw_networkx_labels(G=self.networkx_graph, pos=pos, font_size=font_size)
        nx.draw_networkx_edges(G = self.networkx_graph, pos = pos,
                                alpha=0.4, arrows = False, width=edge_width)
        
        
        plt.title('')
        # plt.ylabel("Latitude", fontsize=15)
        # plt.xlabel("Longitude", fontsize=15)
        fig.tight_layout()
        plt.savefig(rf"C:\Users\20191577\Desktop\network_plots\network_{self.created_with}.pdf", bbox_inches='tight')
        plt.show()
        
    def haversine(self, lonlat1, lonlat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lat1, lon1 = lonlat1
        lat2, lon2 = lonlat2
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r
    
    def summary_statistics(self):
        self.number_of_nodes = nx.number_of_nodes(self.networkx_graph)
        self.number_of_edges = nx.number_of_edges(self.networkx_graph)
        print('\n','####### Summary Statistics #######')
        print(f'Graph created with = {self.path.split("/")[-1]} and {self.created_with}')
        print(f'Number of nodes = {self.number_of_nodes}')
        print(f'Number of edges = {self.number_of_edges}')
        degree_centrality_scores = list(sorted(nx.degree_centrality(self.networkx_graph).items(), key=lambda x : x[1], reverse=True)[:1])
        print(f'The most important node is {degree_centrality_scores[0][0]}({degree_centrality_scores[0][1]:2f})')
        print(f'Number of connected components = {nx.number_connected_components(self.networkx_graph)}')
        # print(f'Average clustering coefficient = {nx.average_clustering(self.networkx_graph):.2f}')
        print(f"Density: {nx.density(self.networkx_graph):.2f}")
        print('#######         END         #######','\n')

    def loglog_degree_histogram(self):
        degree_sequence = sorted((d for n, d in self.networkx_graph.degree()), reverse=True)
        fig = plt.figure(figsize=(3,3))
        # plt.figure(figsize=(3,3))
        plt.loglog(range(1,self.networkx_graph.order()+1),degree_sequence,'k.')
        plt.xlabel('Rank')
        plt.ylabel('Degree')
        fig.tight_layout()
        # plt.savefig(rf"C:\Users\20191577\Desktop\degree_plots\{self.created_with}.pdf")
        plt.show()
        
    def loglog_hist(self):
        degree_sequence = sorted((d for n, d in self.networkx_graph.degree()), reverse=True)
        dmax=max(degree_sequence)
        fig = plt.figure(figsize=(3,3))
        # plt.figure(figsize=(3,3))
        plt.plot(degree_sequence, '#36558f', marker=".")
        # plt.plot(degree_sequence, "k-", marker=".")
        
        plt.title(f"Density: {nx.density(self.networkx_graph):.2f}")
        plt.ylabel("Degree", fontsize=10)
        plt.xlabel("Rank", fontsize=10)
        fig.tight_layout()
        plt.locator_params(axis="both", integer=True, tight=True)
        # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.savefig(rf"C:\Users\20191577\Desktop\degree_plots\{self.created_with}.pdf")
        plt.show()
