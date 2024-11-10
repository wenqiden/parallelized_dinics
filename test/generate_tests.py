import networkx as nx
import random
import uuid

NUM_TO_GENERATE = 2
EDGE_MIN = 0.1
EDGE_MAX = 1
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
        edge_prob = random.uniform(EDGE_MIN, EDGE_MAX)
        G = nx.gnp_random_graph(num_nodes, edge_prob, directed=True)
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        name = str(uuid.uuid4())
        nx.write_edgelist(G, f"{out_dir}/er_network/{name}.edgelist", data=['capacity']) 

def generate_ba_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        num_edge = int(random.uniform(EDGE_MIN, EDGE_MAX) * num_nodes)
        G = nx.barabasi_albert_graph(num_nodes, num_edge)
        G = G.to_directed()
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        name = str(uuid.uuid4())
        nx.write_edgelist(G, f"{out_dir}/ba_network/{name}.edgelist", data=['capacity'])

def generate_grid_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        G = nx.grid_2d_graph(num_nodes, num_nodes, periodic=False)
        G = G.to_directed()
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        name = str(uuid.uuid4())
        nx.write_edgelist(G, f"{out_dir}/grid_network/{name}.edgelist", data=['capacity'])

def generate_small_tests():
    generate_er_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(SMALL_TEST_DIR, NUM_TO_GENERATE, SMALL_NODE_MIN, SMALL_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    
def generate_large_tests():
    generate_er_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(LARGE_TEST_DIR, NUM_TO_GENERATE, LARGE_NODE_MIN, LARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

generate_small_tests()
generate_large_tests()
