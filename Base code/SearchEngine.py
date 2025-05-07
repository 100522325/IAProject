# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement """
    global NODES_EXPANDED
    h = 0
    ...
    NODES_EXPANDED += 1
    return h

def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement """
    global NODES_EXPANDED
    h = 0
    ...
    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors

    # Create a directed graph
    directed_graph = nx.DiGraph()

    # Get dimensions of the detection map
    height, width = detection_map.shape

    # Add all nodes to the graph
    for i in range(height):
        for j in range(width):
            # Create a string identifier for the node: "(i, j)"
            node_id = f"({i}, {j})"
            # Add the node to the graph
            directed_graph.add_node(node_id)

    # Add edges between adjacent nodes, considering the tolerance
    for i in range(height):
        for j in range(width):
            current_node = f"({i}, {j})"

            # Define the possible neighbor coordinates (up, down, left, right)
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

            # Check each potential neighbor
            for i_neighbor, j_neighbor in neighbors:
                # Skip if the neighbor is outside the boundaries
                if i_neighbor < 0 or i_neighbor >= height or j_neighbor < 0 or j_neighbor >= width:
                    continue

                # Get the detection probability of the destination cell
                detection_prob = detection_map[i_neighbor, j_neighbor]

                # Skip if the detection probability exceeds the tolerance
                if detection_prob > tolerance:
                    continue

                # Add the edge with the detection probability as the weight
                neighbor_node = f"({i_neighbor}, {j_neighbor})"
                directed_graph.add_edge(current_node, neighbor_node, weight=detection_prob)


    return directed_graph


def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """Converts coordinates from (lat, lon) into (i, j) """
    # Get the dimensions of the boundaries
    lat_range = boundaries.max_lat - boundaries.min_lat
    lon_range = boundaries.max_lon - boundaries.min_lon

    # Convert each coordinate pair to discrete indices
    discrete_map = []
    for coord in high_level_plan:
        lat, lon = coord

        # Calculate the corresponding indices
        i = int(((lat - boundaries.min_lat) / lat_range) * (map_height - 1))
        j = int(((lon - boundaries.min_lon) / lon_range) * (map_width - 1))

        # Ensure we're within bounds
        i = max(0, min(i, map_height - 1))
        j = max(0, min(j, map_width - 1))

        discrete_map.append((i, j))

    #We create array here because numpy doesn't modify arrays but create new ones.
    return np.array(discrete_map)

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    ...

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    ...
