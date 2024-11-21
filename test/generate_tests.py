import networkx as nx
import os
import random
import uuid
from scipy.io import mmread

NUM_TO_GENERATE = 1
EDGE_PROB_MIN = 0.95
EDGE_PROB_MAX = 0.99
CAPACITY_MIN = 1
CAPACITY_MAX = 20

SMALL_TEST_DIR = "./smalltest_networkx"
SMALL_NODE_MIN = 2
SMALL_NODE_MAX = 128

LARGE_TEST_DIR = "./largetest_networkx"
LARGE_NODE_MIN = 128
LARGE_NODE_MAX = 2048

XLARGE_TEST_DIR = "./extralargetest_networkx"
XLARGE_NODE_MIN = 2048
XLARGE_NODE_MAX = 4096

EXTREME_TEST_DIR = "./extremelargetest_networkx"
EXTREME_NODE_MIN = 4096
EXTREME_NODE_MAX = 8192

IMPOSSIBLE_TEST_DIR = "./impossibletest_networkx"
IMPOSSIBLE_NODE_MIN = 8192
IMPOSSIBLE_NODE_MAX = 16384

DELTAUNAY_TEST_DIR = "./delaunaytest_networkx"

def generate_er_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        edge_prob = random.uniform(EDGE_PROB_MIN, EDGE_PROB_MAX)
        G = nx.gnp_random_graph(num_nodes, edge_prob, directed=True)
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        test_name = str(num_nodes) + "_" + str(num_edges) + "_" + str(uuid.uuid4())
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/er_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
            f.write(f"{source} {sink}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['capacity']}\n")

def generate_ba_test(out_dir, node_num, node_min, node_max, capacity_min, capacity_max):
    for _ in range(node_num):
        num_nodes = random.randint(node_min, node_max)
        num_edge = int(random.uniform(EDGE_PROB_MIN, EDGE_PROB_MAX) * num_nodes)
        if num_edge == 0:
            num_edge = 1 # Barabási–Albert network must have m >= 1
        if num_edge >= num_nodes:
            num_edge = num_nodes - 1
        G = nx.barabasi_albert_graph(num_nodes, num_edge)
        G = G.to_directed()
        for (u, v) in G.edges():
            G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        test_name = str(num_nodes) + "_" + str(num_edges) + "_" + str(uuid.uuid4())
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/ba_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
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
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        test_name = str(num_nodes) + "_" + str(num_edges) + "_" + str(uuid.uuid4())
        source = 0
        sink = random.randint(1, num_nodes - 1)
        with open(f"{out_dir}/grid_network/{test_name}.edgelist", 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
            f.write(f"{source} {sink}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['capacity']}\n")

def generate_delaunay_single_test(G, file_name, capacity_min, capacity_max):
    for (u, v) in G.edges():
        G.edges[u, v]['capacity'] = random.randint(capacity_min, capacity_max)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    test_name = file_name
    source = 0
    sink = random.randint(1, num_nodes - 1)
    with open(f"{DELTAUNAY_TEST_DIR}/{file_name}/{test_name}.edgelist", 'w') as f:
        f.write(f"{num_nodes} {num_edges}\n")
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

def generate_xlarge_tests():
    generate_er_test(XLARGE_TEST_DIR, NUM_TO_GENERATE, XLARGE_NODE_MIN, XLARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(XLARGE_TEST_DIR, NUM_TO_GENERATE, XLARGE_NODE_MIN, XLARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(XLARGE_TEST_DIR, NUM_TO_GENERATE, XLARGE_NODE_MIN, XLARGE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

def generate_extreme_tests():
    generate_er_test(EXTREME_TEST_DIR, NUM_TO_GENERATE, EXTREME_NODE_MIN, EXTREME_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(EXTREME_TEST_DIR, NUM_TO_GENERATE, EXTREME_NODE_MIN, EXTREME_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(EXTREME_TEST_DIR, NUM_TO_GENERATE, EXTREME_NODE_MIN, EXTREME_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

def generate_impossible_tests():
    generate_er_test(IMPOSSIBLE_TEST_DIR, NUM_TO_GENERATE, IMPOSSIBLE_NODE_MIN, IMPOSSIBLE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_ba_test(IMPOSSIBLE_TEST_DIR, NUM_TO_GENERATE, IMPOSSIBLE_NODE_MIN, IMPOSSIBLE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)
    generate_grid_test(IMPOSSIBLE_TEST_DIR, NUM_TO_GENERATE, IMPOSSIBLE_NODE_MIN, IMPOSSIBLE_NODE_MAX, CAPACITY_MIN, CAPACITY_MAX)

def generate_delaunay_tests():
    for file_name in os.listdir(DELTAUNAY_TEST_DIR):
        file_path = os.path.join(DELTAUNAY_TEST_DIR, file_name)
        print(f"Generating tests for {file_name}")
        if os.path.isdir(file_path):
            edgelist_file = None
            for test_files in os.listdir(file_path):
                test_file_path = os.path.join(file_path, test_files)
                if os.path.isfile(test_file_path) and test_files.endswith(".edgelist"):
                    edgelist_file = test_file_path
                    break
            if edgelist_file is not None:
                continue
            mtx_file = None
            for test_files in os.listdir(file_path):
                test_file_path = os.path.join(file_path, test_files)
                if os.path.isfile(test_file_path) and test_files.endswith(".mtx"):
                    mtx_file = test_file_path
                    break
            if mtx_file is None:
                continue

            try:
                matrix = mmread(mtx_file).tocoo()
                G = nx.DiGraph()
                for row, col in zip(matrix.row, matrix.col):
                    if row != col:
                        G.add_edge(col, row)
            except Exception as e:
                print(f"Failed to read {mtx_file}")
                G = nx.DiGraph()
                # try to load it line by line
                with open(mtx_file, 'r') as f:
                    lines = f.readlines()
                    G = nx.DiGraph()
                    for line in lines[3:]:
                        try:
                            u, v = map(int, line.split())
                            if u != v:
                                G.add_edge(u, v)
                                G.add_edge(v, u)
                        except:
                            try:
                                u = int(line.split()[0].strip())
                                v = int(line.split()[1].strip())
                                if u != v:
                                    G.add_edge(u, v)
                                    G.add_edge(v, u)
                            except:
                                continue
            generate_delaunay_single_test(G, file_name, CAPACITY_MIN, CAPACITY_MAX)
            

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

if not os.path.exists(XLARGE_TEST_DIR):
    os.makedirs(XLARGE_TEST_DIR)
    os.makedirs(f"{XLARGE_TEST_DIR}/er_network")
    os.makedirs(f"{XLARGE_TEST_DIR}/ba_network")
    os.makedirs(f"{XLARGE_TEST_DIR}/grid_network")

if not os.path.exists(EXTREME_TEST_DIR):
    os.makedirs(EXTREME_TEST_DIR)
    os.makedirs(f"{EXTREME_TEST_DIR}/er_network")
    os.makedirs(f"{EXTREME_TEST_DIR}/ba_network")
    os.makedirs(f"{EXTREME_TEST_DIR}/grid_network")


if not os.path.exists(IMPOSSIBLE_TEST_DIR):
    os.makedirs(IMPOSSIBLE_TEST_DIR)
    os.makedirs(f"{IMPOSSIBLE_TEST_DIR}/er_network")
    os.makedirs(f"{IMPOSSIBLE_TEST_DIR}/ba_network")
    os.makedirs(f"{IMPOSSIBLE_TEST_DIR}/grid_network")

if not os.path.exists(DELTAUNAY_TEST_DIR):
    os.makedirs(DELTAUNAY_TEST_DIR)

# generate_small_tests()
# generate_large_tests()
# generate_xlarge_tests()
# generate_extreme_tests()
generate_impossible_tests()
# generate_delaunay_tests()
