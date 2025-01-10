import heapq
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def beam_search_subgraph(G, start_node, n, beam_width=5):
    if start_node not in G or n <= 0 or n > G.number_of_nodes():
        return None, float('-inf')
    
    if n == 1:
        return {start_node}, 0.0

    def subgraph_weight(nodes):
        sub = G.subgraph(nodes)
        return sum(d["weight"] for _, _, d in sub.edges(data=True))


    initial_weight = 0.0
    initial_frontier = set()
    for nbr in G.neighbors(start_node):
        if nbr != start_node:
            w = G[start_node][nbr].get("weight", 0)
            initial_frontier.add((w, start_node, nbr))

    beam = [(-initial_weight, frozenset([start_node]), frozenset(initial_frontier))]

    best_subgraph = None
    best_score = float('-inf')

    # We'll expand until we reach subgraphs of size n
    while beam:
        # Next-level expansions (candidates)
        candidates = []

        # Pop expansions from the current beam
        for (neg_score, nodes_fo, frontier_fo) in beam:
            current_nodes = set(nodes_fo)
            current_frontier = set(frontier_fo)
            current_score = -neg_score

            if len(current_nodes) == n:
                # We have a candidate subgraph of the desired size
                if current_score > best_score:
                    best_score = current_score
                    best_subgraph = current_nodes
                # We won't expand further this subgraph
                continue

            # Expand this partial subgraph by adding a new node from frontier
            # We consider each frontier edge that leads to a new node
            for (w, u, v) in current_frontier:
                if v not in current_nodes:
                    # This edge would add node 'v' to the subgraph
                    new_nodes = current_nodes | {v}
                    new_score = subgraph_weight(new_nodes)

                    new_frontier = set()
                    for (w2, x2, y2) in current_frontier:
                        # If y2 is not in new_nodes, this edge is still "frontier"
                        if (x2 in new_nodes and y2 not in new_nodes) or \
                           (y2 in new_nodes and x2 not in new_nodes):
                            new_frontier.add((w2, x2, y2))
                    
                    # Add new edges from the newly added node 'v'
                    for nbr in G.neighbors(v):
                        if nbr not in new_nodes:
                            w_edge = G[v][nbr].get("weight", 0)
                            new_frontier.add((w_edge, v, nbr))

                    # Add candidate expansion
                    candidates.append((-new_score, frozenset(new_nodes), frozenset(new_frontier)))

        if not candidates:
            # No more expansions possible
            break

        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    # Return final best
    return best_subgraph, best_score

def plot_num_nodes_vs_weight_proportion(json_file_path_1, json_file_path_2):

    with open(json_file_path_1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(json_file_path_2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    total_weight_1 = data1[-1]["weight"]
    total_weight_2 = data2[-1]["weight"]

    x1 = [entry["num_nodes"] for entry in data1]
    y1 = [entry["weight"] / total_weight_1 for entry in data1]

    x2 = [entry["num_nodes"] for entry in data2]
    y2 = [entry["weight"] / total_weight_2 for entry in data2]

    # Find the first occurrence of "Fractured Marika" for each file
    fractured_index_1 = None
    for i, entry in enumerate(data1):
        if "Fractured Marika" in entry["maximal_comp"]:
            fractured_index_1 = i
            break

    fractured_index_2 = None
    for i, entry in enumerate(data2):
        if "Fractured Marika" in entry["maximal_comp"]:
            fractured_index_2 = i
            break

    plt.figure(figsize=(8, 5))

    plt.plot(x1, y1, marker='o', markersize=2, label='Boss Network')
    plt.plot(x2, y2, marker='o', markersize=2, label='Item Network')

    if fractured_index_1 is not None:
        plt.plot(
            x1[fractured_index_1],
            y1[fractured_index_1],
            marker='X', markersize=10, color='blue',
            label='Fractured Marika in Boss Network'
        )

    if fractured_index_2 is not None:
        plt.plot(
            x2[fractured_index_2],
            y2[fractured_index_2],
            marker='X', markersize=10, color='orange',
            label='Fractured Marika in Item Network'
        )

    plt.xlabel("number of nodes in maximal component")
    plt.ylabel("Proportion of weight in maximal component")
    plt.title("Comparing the progression of boss and item value accumulation over increasing maximally weighted components")
    plt.legend()
    plt.grid(True)
    plt.show()

    x1_np = np.array(x1).reshape(-1, 1)
    y1_np = np.array(y1)
    model1 = LinearRegression().fit(x1_np, y1_np)
    r2_1 = model1.score(x1_np, y1_np)
    print(f"File 1 - Linear fit R^2: {r2_1:.4f}, Coefficient: {model1.coef_[0]:.4f}")

    x2_np = np.array(x2).reshape(-1, 1)
    y2_np = np.array(y2)
    model2 = LinearRegression().fit(x2_np, y2_np)
    r2_2 = model2.score(x2_np, y2_np)
    print(f"File 2 - Linear fit R^2: {r2_2:.4f}, Coefficient: {model2.coef_[0]:.4f}")

def plot_comp_intersection_union(json_file_path_1, json_file_path_2):

    with open(json_file_path_1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(json_file_path_2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    x_vals = []
    intersection_vals = []
    union_vals = []

    for i in range(len(data1)):
        comp1 = data1[i]["maximal_comp"]
        comp2 = data2[i]["maximal_comp"]

        # Convert to sets
        set1 = set(comp1)
        set2 = set(comp2)

        # Intersection and union
        inter = set1.intersection(set2)
        uni = set1.union(set2)

        num_nodes = data1[i]["num_nodes"]

        # Normalize by node number (308)
        intersection_size_norm = len(inter) / 308
        union_size_norm = len(uni) / 308

        x_vals.append(num_nodes)
        intersection_vals.append(intersection_size_norm)
        union_vals.append(union_size_norm)

    plt.figure(figsize=(8,5))

    plt.plot(x_vals, intersection_vals, marker='o', markersize=2, label='Intersection')
    plt.plot(x_vals, union_vals, marker='o', markersize=2, label='Union')

    y_line = [x / 308 for x in x_vals]
    plt.plot(x_vals, y_line, linestyle='--', color='gray', label='Reference (x/308)')


    plt.xlabel("number of nodes in maximal component")
    plt.ylabel("Proportion of total nodes")
    plt.title("Intersection and Union Proportions of Maximal Components from Boss and Item Networks")
    plt.legend()
    plt.grid(True)
    plt.show()