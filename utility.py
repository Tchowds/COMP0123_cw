import json
from collections import defaultdict, deque
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def construct_graph(config):
    graph = defaultdict(list)
    for entry in config.get("map", []):
        site = entry.get("site")
        edges = entry.get("edges", [])
        graph[site].extend(edges)
    return graph

def verify_names(config, graph):
    sites = set(config.get("sites of grace", []))
    for site, edges in graph.items():
        if site not in sites:
            print(f"Invalid site name found: {site}")
            return True
        for edge in edges:
            if edge not in sites:
                print(f"Invalid edge name '{edge}' found in site '{site}'")
                return True

def is_fully_connected(graph):
    if not graph:
        return True, []
    visited = set()
    queue = deque()
    start = next(iter(graph))
    queue.append(start)
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            queue.extend([neighbor for neighbor in graph[current] if neighbor not in visited])
    unreachable = list(set(graph.keys()) - visited)
    return len(unreachable) == 0, unreachable


def print_shortest_path(graph, start, end):

    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current, path = queue.popleft()
        if current == end:
            print(" -> ".join(path))
            return
        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    print(f"No path found from {start} to {end}.")

def find_diameter(graph):

    def bfs(start):
        visited = {start}
        queue = deque([(start, [start])])
        farthest_node = start
        path = []
        while queue:
            current, current_path = queue.popleft()
            farthest_node = current
            path = current_path
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_path + [neighbor]))
        return len(path) - 1, path

    diameter = 0
    diameter_path = []
    for node in graph:
        length, path = bfs(node)
        if length > diameter:
            diameter = length
            diameter_path = path
    print("Diameter of the graph is", diameter)
    print("Nodes on the diameter path:", " -> ".join(diameter_path))

def plot_degree_distribution(graph):
    degree_count = defaultdict(int)
    for node, neighbors in graph.items():
        degree = len(neighbors)
        degree_count[degree] += 1

    degrees = sorted(degree_count.keys())
    counts = [degree_count[deg] for deg in degrees]

    plt.figure(figsize=(10, 6))
    plt.bar(degrees, counts, color='skyblue')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.xticks(degrees)
    plt.show()

def clustering_coefficient(graph, node):
    neighbors = set(graph[node])
    if len(neighbors) < 2:
        coefficient = 0.0
    else:
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        actual = 0
        for neighbor in neighbors:
            neighbor_neighbors = set(graph[neighbor])
            actual += len(neighbors.intersection(neighbor_neighbors))
        actual = actual / 2
        coefficient = actual / possible
    print(f"Clustering coefficient of node {node}: {coefficient}")

def average_clustering_coefficient(graph):
    total = 0
    count = 0
    for node in graph:
        neighbors = set(graph[node])
        if len(neighbors) < 2:
            cc = 0.0
        else:
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            actual = 0
            for neighbor in neighbors:
                neighbor_neighbors = set(graph[neighbor])
                actual += len(neighbors.intersection(neighbor_neighbors))
            actual = actual / 2
            cc = actual / possible
        total += cc
        count += 1
    average = total / count if count > 0 else 0
    print(f"Average clustering coefficient: {average}")

def rich_club_coefficient(graph):
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}
    max_degree = max(degrees.values()) if degrees else 0
    rich_club_coefs = {}
    normalized_coefs = {}

    for k in range(1, max_degree + 1):
        rich_nodes = [node for node, degree in degrees.items() if degree > k]
        E = 0
        for i, node in enumerate(rich_nodes):
            for neighbor in rich_nodes[i+1:]:
                if neighbor in graph[node]:
                    E += 1
        N = len(rich_nodes)
        possible = N * (N - 1) / 2
        coef = E / possible if possible > 0 else 0
        rich_club_coefs[k] = coef

    for k, coef in rich_club_coefs.items():
        print(f"Rich club coefficient for k={k}: {coef}")

def plot_3d_graph(graph):


    counts = defaultdict(int)
    for node, neighbors in graph.items():
        x = len(neighbors)
        if neighbors:
            avg_deg = sum(len(graph[n]) for n in neighbors) / len(neighbors)
        else:
            avg_deg = 0
        counts[(x, avg_deg)] += 1

    xs, ys, zs = [], [], []
    for (deg, avg_deg), count in counts.items():
        xs.append(deg)
        ys.append(avg_deg)
        zs.append(count)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z in zip(xs, ys, zs):
        ax.plot([x, x], [y, y], [0, z], color='blue')
    ax.scatter(xs, ys, zs, color='red')

    # Draw a plane covering all points
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    plane_x, plane_y = np.meshgrid(np.linspace(min_x, max_x, 10),
                                   np.linspace(min_y, max_y, 10))
    plane_z = np.full_like(plane_x, max(zs))
    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.2)

    ax.set_xlabel('Degree')
    ax.set_ylabel('Avg Neighbor Degree')
    ax.set_zlabel('Cumulative Count')
    plt.show()

def verify_bidirectional_edges(graph):
    missing_edges = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if node not in graph[neighbor]:
                missing_edges.append((neighbor, node))
    if missing_edges:
        print("Missing bidirectional edges:")
        for edge in missing_edges:
            print(f"Edge from {edge[0]} to {edge[1]} is missing.")
    else:
        print("All edges are bidirectional.")


def verify_undirected_edges(config):
    """
    Verify that all edges in the 'map' array of the config object define
    an undirected graph correctly and that corresponding 'edge_enemies'
    arrays match in both directions.

    :param config: A dictionary loaded from JSON, e.g. config = json.load(...)
    """

    # Collect sites into a dictionary for faster lookup:
    # { site_name: { "edges": [...], "edge_enemies": [...], ... }, ... }
    site_dict = {}
    for site_info in config["map"]:
        site_name = site_info["site"]
        site_dict[site_name] = site_info

    # Helper to retrieve edge index of 'dest_site' in the edges of 'site_info'
    def get_edge_index(site_info, dest_site):
        """Return the index of dest_site in site_info['edges'] or -1 if not found."""
        edges = site_info["edges"]
        try:
            return edges.index(dest_site)
        except ValueError:
            return -1

    all_valid = True

    # Iterate over each site and each edge to verify
    for site_info in config["map"]:
        site_name = site_info["site"]
        # print(site_name)
        edges = site_info["edges"]
        edge_enemies = site_info["edge_items"]

        for i, destination_site in enumerate(edges):
            # print(destination_site)
            enemies_for_edge = edge_enemies[i]

            # 1. Check if destination site exists
            if destination_site not in site_dict:
                print(f"[ERROR] '{destination_site}' referenced by '{site_name}' does not exist in the map.")
                all_valid = False
                continue  # Nothing more to compare if the site doesn't exist

            # 2. Check if destination site references back this site
            dest_site_info = site_dict[destination_site]
            j = get_edge_index(dest_site_info, site_name)

            if j == -1:
                # Both directions don't exist if site A -> site B but site B doesn't have site A
                print(f"[ERROR] '{destination_site}' does not have an edge back to '{site_name}'.")
                all_valid = False
                continue

            # 3. Check if the enemies sub-array matches
            dest_enemies = dest_site_info["edge_items"][j]
            if enemies_for_edge != dest_enemies:
                print(
                    f"[ERROR] Enemies mismatch on edge '{site_name}' <-> '{destination_site}'.\n"
                    f"  '{site_name}' has enemies {enemies_for_edge}\n"
                    f"  '{destination_site}' has enemies {dest_enemies}"
                )
                all_valid = False

    if all_valid:
        print("All edges are valid and consistent.")


def main():
    config = load_config('config.json')
    graph = construct_graph(config)
    verify_undirected_edges(config)
    
    if verify_names(config, graph):
        print("Invalid location names found.")
        return
    
    verify_bidirectional_edges(graph)
    connected, unreachable = is_fully_connected(graph)
    if connected:
        print("The graph is fully connected.")
    else:
        print("The graph is not fully connected.")
        print("Unreachable nodes:", unreachable)


    # print_shortest_path(graph, "The First Step", "Fractured Marika")
    # plot_degree_distribution(graph)
    # find_diameter(graph)
    # clustering_coefficient(graph, "Avenue Balcony")
    # average_clustering_coefficient(graph)
    # rich_club_coefficient(graph)
    # plot_3d_graph(graph)


if __name__ == "__main__":
    main()

'''
Changes
- Removed link from Divine bridge to manor first floor
- Varre link to Mohg removed
- Frenzied flame link to deeproot removed (may need to add back)
- Temporarily added one way link from Raya Lucaria Grand Library to Main academy gate
'''