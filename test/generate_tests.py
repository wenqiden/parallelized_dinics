import networkx as nx
import os
import random
import uuid

NUM_TO_GENERATE = 10
EDGE_PROB_MIN = 0.1
EDGE_PROB_MAX = 0.5
CAPACITY_MIN = 1
CAPACITY_MAX = 20

SMALL_TEST_DIR = "./smalltest_networkx"
SMALL_NODE_MIN = 2
SMALL_NODE_MAX = 128

LARGE_TEST_DIR = "./largetest_networkx"
LARGE_NODE_MIN = 128
LARGE_NODE_MAX = 1024

def generate_er_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        edge_prob = random.uniform(EDGE_PROB_MIN, EDGE_PROB_MAX)
        G = nx.gnp_random_graph(num_nodes, edge_prob, directed=True)
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        test_name = str(uuid.uuid4())

        num_nodes = G.number_of_nodes()
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/er_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {G.number_of_edges()}\n")
            f.write(f"{source} {sink}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['capacity']}\n")

def generate_ba_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        num_edge = int(random.uniform(EDGE_PROB_MIN, EDGE_PROB_MAX) * num_nodes)
        if num_edge == 0:
            num_edge = 1 # Barabási–Albert network must have m >= 1
        G = nx.barabasi_albert_graph(num_nodes, num_edge)
        G = G.to_directed()
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        test_name = str(uuid.uuid4())

        num_nodes = G.number_of_nodes()
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/ba_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {G.number_of_edges()}\n")
            f.write(f"{source} {sink}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['capacity']}\n")

def generate_grid_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        dimx = int(num_nodes ** 0.5)
        dimy = num_nodes // dimx
        G = nx.grid_2d_graph(dimx, dimy, periodic=False)
        G = G.to_directed()

        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, node_mapping)

        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        test_name = str(uuid.uuid4())

        num_nodes = G.number_of_nodes()
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/grid_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {G.number_of_edges()}\n")
            f.write(f"{source} {sink}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['capacity']}\n")


def generate_small_tests():
    generate_er_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

def generate_large_tests():
    generate_er_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

if not os.path.exists(SMALL_TEST_DIR):
    os.makedirs(SMALL_TEST_DIR)
    os.makedirs(f"{SMALL_TEST_DIR}/er_network")
    os.makedirs(f"{SMALL_TEST_DIR}/ba_network")
    os.makedirs(f"{SMALL_TEST_DIR}/grid_network")

if not os.path.exists(LARGE_TEST_DIR):
    os.makedirs(LARGE_TEST_DIR)
    os.makedirs(f"{LARGE_TEST_DIR}/er_network")
    os.makedirs(f"{LARGE_TEST_DIR}/ba_network")
    os.makedirs(f"{LARGE_TEST_DIR}/grid_network")

generate_small_tests()
generate_large_tests()