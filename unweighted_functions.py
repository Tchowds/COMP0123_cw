import networkx as nx
import random
import matplotlib.pyplot as plt


def get_top_k_degree_nodes(G, k):
    degrees = G.degree()
    sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    top_k_nodes = sorted_degrees[:k]
    for node, degree in top_k_nodes:
        print(f"Node {node}: Degree {degree}")
    return top_k_nodes

def get_top_k_betweenness_centrality_nodes(G, k):
    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    top_k_nodes = sorted_betweenness[:k]
    for node, centrality in top_k_nodes:
        print(f"Node {node}: Betweenness Centrality {centrality}")
    return top_k_nodes


def random_traversal(
    G: nx.Graph,
    start_node,
    end_node,
    setA,
    setB,
    seed=None
):
    # Optionally set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Extract node names (ignore second element) and convert to set
    setA_nodes = {item[0] for item in setA}
    setB_nodes = {item[0] for item in setB}

    # Basic checks
    if start_node not in G:
        raise ValueError(f"start_node '{start_node}' is not in the graph.")
    if end_node not in G:
        raise ValueError(f"end_node '{end_node}' is not in the graph.")
    if len(setA_nodes) == 0 or len(setB_nodes) == 0:
        raise ValueError("One of the sets is empty; cannot compute proportions.")

    visited_nodes = {start_node}
    used_edges = set()    # store edges in canonical form (min(u,v), max(u,v))
    current_node = start_node

    iteration_list = []
    propA_list = []
    propB_list = []

    iteration_count = 0

    # Track the proportion for iteration 0 (before any move)
    iteration_list.append(iteration_count)
    propA_list.append(len(visited_nodes.intersection(setA_nodes)) / len(setA_nodes))
    propB_list.append(len(visited_nodes.intersection(setB_nodes)) / len(setB_nodes))

    while current_node != end_node:
        # Collect all edges from visited_nodes to anywhere, excluding used edges
        possible_edges = []
        for u in visited_nodes:
            for v in G[u]:
                e = (u, v) if u < v else (v, u)
                if e not in used_edges:
                    possible_edges.append(e)

        if not possible_edges:
            break

        chosen_edge = random.choice(possible_edges)
        used_edges.add(chosen_edge)
        iteration_count += 1

        (a, b) = chosen_edge
        if current_node == a:
            next_node = b
        elif current_node == b:
            next_node = a
        else:
            if a not in visited_nodes:
                next_node = a
            else:
                next_node = b

        # Add the next_node to visited_nodes
        visited_nodes.add(next_node)
        current_node = next_node

        # Track and store proportions for this iteration
        iteration_list.append(iteration_count)
        propA_list.append(len(visited_nodes.intersection(setA_nodes)) / len(setA_nodes))
        propB_list.append(len(visited_nodes.intersection(setB_nodes)) / len(setB_nodes))

        # Check if we've reached the end node
        if current_node == end_node:
            break

    return visited_nodes, iteration_count, propA_list, propB_list, iteration_list

def multiple_runs_random_traversal_overlay(
    G: nx.Graph,
    start_node,
    end_node,
    setA,
    setB,
    num_runs=10,
    seed=None
):

    # Initialize figure
    plt.figure(figsize=(8, 5))

    all_data = []

    for i in range(num_runs):
        # Optionally vary the seed per run to get different random outcomes
        run_seed = (seed + i) if seed is not None else None

        visited_nodes, it_count, propA_list, propB_list, it_list = random_traversal(
            G, start_node, end_node, setA, setB, seed=run_seed
        )

        print(f"Run {i+1}/{num_runs}: "
              f"steps={it_count}, finalA={propA_list[-1]:.2f}, finalB={propB_list[-1]:.2f}, "
              f"visited={len(visited_nodes)}")

        plt.scatter(it_list, propA_list, color='blue', alpha=0.5, s=2, marker='x')
        plt.scatter(it_list, propB_list, color='red', alpha=0.5, s=2, marker='x')

        # Store the data if we want to analyze it later
        all_data.append((propA_list, propB_list, it_list, visited_nodes, it_count))

    # Add labels/legend
    plt.xlabel("Iterations")
    plt.ylabel("Proportion Visited")
    plt.title(f"Random Traversal of the Graph: Proportion of Highest Degree Nodes and Highest Betweenness Nodes Visited")
    plt.grid(True)

    plt.scatter([], [], color='blue', label='Proportion of top 30 highest-degree nodes visited')
    plt.scatter([], [], color='red', label='Proportion of top 30 highest-betweenness nodes visited')
    plt.legend()

    plt.show()

    return all_data