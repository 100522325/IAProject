# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0


def h1(current_node, objective_node) -> np.float32:
    """
    Heuristic 1: Manhattan Distance with Minimum Cost (MDMC)
    Estimates cost assuming Manhattan distance with minimum cost per step (EPSILON)
    """
    global NODES_EXPANDED
    # Parse current and objective node coordinates from string format "(i, j)"
    current_coords = eval(current_node)
    objective_coords = eval(objective_node)

    # Calculate Manhattan distance: |r_curr - r_goal| + |c_curr - c_goal|
    manhattan_distance = abs(current_coords[0] - objective_coords[0]) + abs(current_coords[1] - objective_coords[1])

    # Multiply by minimum cost (EPSILON)
    h = manhattan_distance * EPSILON

    NODES_EXPANDED += 1
    return h
def h2(current_node, objective_node) -> np.float32:
    """
    Heuristic 0: Zero Heuristic
    Always returns 0 regardless of current and goal position
    """
    global NODES_EXPANDED
    h = 0.0
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
    # Reset the global counter for nodes expanded
    global NODES_EXPANDED
    NODES_EXPANDED = 0

    # Convert the POI coordinates from (lat, lon) to grid coordinates (i, j)
    discrete_locations = discretize_coords(
        high_level_plan=locations,
        boundaries=boundaries,
        map_width=map_width,
        map_height=map_height
    )

    # Create a container for the solution plan
    solution_plan = []

    # Set the initial POI as the starting point
    current_index = initial_location_index

    # Visit all POIs in sequence (high-level planning)
    num_locations = len(discrete_locations)
    for i in range(num_locations):
        # Get the next location index (circular if needed)
        next_index = (current_index + 1) % num_locations

        # Get the grid coordinates for current and next POIs
        current_point = discrete_locations[current_index]
        next_point = discrete_locations[next_index]

        # Convert to node IDs used in the graph
        source = f"({current_point[0]}, {current_point[1]})"
        target = f"({next_point[0]}, {next_point[1]})"

        try:
            # Use A* algorithm from networkx to find the path
            path = nx.astar_path(
                G=G,
                source=source,
                target=target,
                heuristic=heuristic_function,
                weight='weight'
            )

            # Add the found path to the solution
            solution_plan.append(path)

            # Update current index for next iteration
            current_index = next_index

        except nx.NetworkXNoPath:
            # Handle case where no path exists
            print(f"No path found between {source} and {target}. Check tolerance value or map configuration.")
            return [], NODES_EXPANDED

    return solution_plan, NODES_EXPANDED

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    total_cost = 0.0

    # Iterate through each segment of the plan
    for path in solution_plan:
        # Calculate the cost of each segment by summing edge weights
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Get the weight of the edge (detection probability)
            edge_weight = G.edges[source, target]['weight']
            total_cost += edge_weight

    return total_cost
