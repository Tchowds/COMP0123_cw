# COMP0123 Complex Networks Coursework

## Project Overview

This repository contains coursework tools for the COMP0123 Complex Networks Module at UCL. The coursework involves analyzing and modeling complex networks using various algorithms and techniques.


## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Tchowds/COMP0123_cw.git
cd COMP0123_cw
pip install -r requirements.txt
```

## Usage

Run the main script to perform network analysis:

```bash
python main.py
```

Below is a breakdown of each file and its primary role within the project.

---

## `main.py`
- **Purpose**:  
  - **Builds three different graph representations** from a single JSON source:
    1. **Unweighted graph (`G1`)**  
    2. **Enemy-weighted graph (`G2`)**  
    3. **Item-weighted graph (`G3`)**  
  - **Demonstrates usage** of core functions by:
    - Identifying top degree nodes and top betweenness-centrality nodes in the **unweighted** graph.
    - Running multiple random traversals over the unweighted graph, tracking how many of these “important” nodes get visited.
    - (Optionally) calling `generate_sub_graph` to perform beam searches on the **enemy-weighted** and **item-weighted** graphs, and outputting the results to JSON.
    - Visualizing:
      - How subgraph size relates to total weight (enemy or item).  
      - Intersection and union of subgraphs between the two weighted graphs.

- **Key Functions**:
  - **`build_three_graphs_from_json(json_file_path)`**:  
    Reads a config JSON to build and return three `networkx.Graph` objects.  
  - **`generate_sub_graph(graph, filename)`**:  
    Uses beam search (from `weighted_functions`) to find maximal components of increasing size, storing the results in a JSON file.  

---

## `unweighted_functions.py`
- **Purpose**:  
  - Contains **analysis and traversal functions** for **unweighted** graphs.
  - Focuses on node centrality metrics and random traversal logic.

- **Key Functions**:
  1. **`get_top_k_degree_nodes(G, k)`**:  
     Returns the top-`k` highest-degree nodes.  
  2. **`get_top_k_betweenness_centrality_nodes(G, k)`**:  
     Returns the top-`k` nodes by betweenness centrality.  
  3. **`random_traversal(G, start_node, end_node, setA, setB, seed=None)`**:  
     Executes a random walk from `start_node` to `end_node`, tracking how many nodes in two sets (`setA`, `setB`) get discovered as the walk proceeds.  
  4. **`multiple_runs_random_traversal_overlay(G, start_node, end_node, setA, setB, num_runs=10, seed=None)`**:  
     Performs multiple random traversals and overlays the results in a single plot to compare the discovery rates of “important” nodes (e.g., top-degree vs. top-betweenness).  

---

## `weighted_functions.py`
- **Purpose**:  
  - Provides **search, analytics, and visualization** functions for **weighted** graphs (enemy-weighted and item-weighted).

- **Key Functions**:
  1. **`beam_search_subgraph(G, start_node, n, beam_width=5)`**:  
     A beam search procedure to find a subgraph of size `n` that maximizes the total edge weight.  
     - Designed for iterative expansion from a `start_node`.  
     - Returns the best subgraph found and its total weight.  
  2. **`plot_num_nodes_vs_weight_proportion(json_file_path_1, json_file_path_2)`**:  
     Reads subgraph data from two JSON files (boss and item). Plots how the proportion of total weight scales with the number of nodes. Also fits a linear regression to measure the relationship between component size and accumulated weight.  
  3. **`plot_comp_intersection_union(json_file_path_1, json_file_path_2)`**:  
     Compares corresponding subgraphs between two JSON datasets (boss vs. item) at each `num_nodes` step, plotting the **intersection** and **union** sizes (as proportions of the entire node set).

---
