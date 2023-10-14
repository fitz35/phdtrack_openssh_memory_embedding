import networkx as nx
import glob
import os
import pickle
import time
from embedding_deep_learning.params.params import ProgramParams
from torch_geometric.utils import from_networkx, convert
import torch
from multiprocessing import Pool

def graph_cleaning(nx_graph: nx.Graph):
    """
    Clean the graph from all attributes.
    """
    def set_label(node):
        nx_graph.nodes[node]['label'] = 1 if 'KN_KEY' in node else 0

    # remove all attributes from nodes
    for node in nx_graph.nodes():
        nx_graph.nodes[node].clear()
    
        set_label(node)

    return nx_graph

def convert_graph_to_ml_data(nx_graph: nx.Graph):
    """
    Convert the given NetworkX graph to a PyTorch Geometric data object.
    """
    return from_networkx(nx_graph)

def load_annotated_graph(
    params: ProgramParams ,
    annotated_graph_dot_gv_file_path: str
):
    """
    Load annotated graph from given path.
    Use pickle to save the graph to a file or load it from a file 
    if it already exists.
    Perform graph cleaning.
    Convert the graph to a PyTorch Geometric data object.
    """    
    # load annotated graph
    file_name = os.path.basename(annotated_graph_dot_gv_file_path)
    nx_graph_pickle_path = params.PICKLE_DATASET_DIR_PATH + "/" + file_name + ".pickle"

    # Check if the save file exists
    if os.path.exists(nx_graph_pickle_path):
        # Load the NetworkX graph from the save file
        with open(nx_graph_pickle_path, 'rb') as file:
            data = pickle.load(file)
    else:
        nx_graph = nx.Graph(nx.nx_pydot.read_dot(annotated_graph_dot_gv_file_path))
        # cleaning
        nx_graph = graph_cleaning(nx_graph)

        # convert to PyTorch Geometric data object
        data = convert_graph_to_ml_data(nx_graph)

        # Save the NetworkX graph to a file using pickle
        with open(nx_graph_pickle_path, 'wb') as file:
            pickle.dump(nx_graph, file)
    
    
    return data

def dev_load_training_graphs(
    params: ProgramParams ,
    annotated_graph_dot_gv_dir_path: str
):
    """
    Load all annotated graphs from given path.
    """
    # get all files in the folder
    annotated_graph_dot_gv_file_paths = glob.glob(annotated_graph_dot_gv_dir_path + "/*.gv")

    # for now, as a test, filter only "Training" graphs
    annotated_graph_dot_gv_file_paths = [annotated_graph_dot_gv_file_path for annotated_graph_dot_gv_file_path in annotated_graph_dot_gv_file_paths if "Training" in annotated_graph_dot_gv_file_path]
    # for now, only load 32 graphs
    annotated_graph_dot_gv_file_paths = annotated_graph_dot_gv_file_paths[:32]
    print("Loading " + str(len(annotated_graph_dot_gv_file_paths)) + " graphs...")

    # Parallelize the loading of the graphs into data objects
    # Parallelize the loading of the graphs into data objects
    with Pool() as pool:
        datas = pool.starmap(load_annotated_graph, [(params, path) for path in annotated_graph_dot_gv_file_paths])
    
    return datas