import json
import networkx as nx
import matplotlib.pyplot as plt

import unweighted_functions as uf
import weighted_functions as wf

def build_three_graphs_from_json(json_file_path):
    """
    Reads the specified JSON file and returns three undirected graphs:

      1) G1: Unweighted graph
      2) G2: Graph weighted by the sum of enemy rankings on each edge
      3) G3: Graph weighted by the sum of item rankings on each edge

    Parameters
    ----------
    json_file_path : str
        The path to the JSON file containing 'map', 'enemy_rankings', and 'item_rankings'.

    Returns
    -------
    tuple
        (G1, G2, G3) where each is a networkx.Graph instance.
    """

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the ranking dictionaries
    enemy_rankings = data["enemy_rankings"]
    item_rankings  = data["item_rankings"]

    # Initialize three undirected graphs
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()

    # First, add all nodes
    for node in data["map"]:
        site_name = node["site"]
        # Make sure each node exists in all graphs
        G1.add_node(site_name)
        G2.add_node(site_name)
        G3.add_node(site_name)

    # Now, add edges
    for node in data["map"]:
        site_name   = node["site"]
        edges       = node["edges"]
        edge_enemies = node.get("edge_enemies", [])
        edge_items   = node.get("edge_items", [])

        # Iterate over the edges and their corresponding "edge_enemies" / "edge_items"
        for i, adjacent_site in enumerate(edges):
            # Unweighted graph (G1) - just add the edge
            G1.add_edge(site_name, adjacent_site)

            # Compute total enemy weight for this edge
            enemies_on_edge = edge_enemies[i] if i < len(edge_enemies) else []
            enemy_weight = sum(enemy_rankings.get(enemy, 0) for enemy in enemies_on_edge)
            # Add weighted edge to G2
            G2.add_edge(site_name, adjacent_site, weight=enemy_weight)

            # Compute total item weight for this edge
            items_on_edge = edge_items[i] if i < len(edge_items) else []
            item_weight = sum(item_rankings.get(item, 0) for item in items_on_edge)
            # Add weighted edge to G3
            G3.add_edge(site_name, adjacent_site, weight=item_weight)

    return G1, G2, G3


def generate_sub_graph(graph, filename):

    results = []
    for i in range(1, graph.number_of_nodes() + 1):
        print(f"Processing {i} nodes")
        maximal_comp, weight = wf.beam_search_subgraph(graph, "Stranded Graveyard", i, 25)
        results.append({
            "num_nodes": i,
            "maximal_comp": list(maximal_comp),
            "weight": weight
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    base_graph, boss_graph, item_graph = build_three_graphs_from_json("config.json")


    # Top 30 highest degree nodes
    highest_degree_nodes = uf.get_top_k_degree_nodes(base_graph, 30)

    # Top 30 highest betweenness centrality nodes
    highest_betweenness_nodes = uf.get_top_k_betweenness_centrality_nodes(base_graph, 30)

    # Random traversals with overlay
    uf.multiple_runs_random_traversal_overlay(
        base_graph, "Stranded Graveyard", "Fractured Marika", highest_degree_nodes, highest_betweenness_nodes
    )

    # Generate maximal components and store in JSON (takes very long time)

    # generate_sub_graph(boss_graph, "boss_graph.json")
    # generate_sub_graph(item_graph, "item_graph.json")

    # Plot number of nodes vs. weight proportion
    wf.plot_num_nodes_vs_weight_proportion("boss_graph.json", "item_graph.json")

    # Plot intersection and union of maximal components
    wf.plot_comp_intersection_union("boss_graph.json", "item_graph.json")
