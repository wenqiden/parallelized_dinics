import subprocess, time, filecmp, os
import sys, traceback
import networkx as nx

SMALL_TEST_DIR = "./smalltest_networkx"
SOL_PATH = "../SequentialDinics.cpp"

# compile the solution
if os.path.exists("sol"):
    subprocess.run(["rm", "sol"])
subprocess.run(["g++", SOL_PATH, "-o", "sol"])

for file_name in os.listdir(SMALL_TEST_DIR):
    file_path = os.path.join(SMALL_TEST_DIR, file_name)
    if os.path.isdir(file_path):
        print("="*20)
        print(f"Testing {file_name}")
        for test_files in os.listdir(file_path):
            test_file_path = os.path.join(file_path, test_files)
            if os.path.isfile(test_file_path):
                try:
                    # load the graph
                    with open(test_file_path, 'r') as f:
                        num_nodes, num_edges = map(int, f.readline().split())
                        source, sink = map(int, f.readline().split())
                        G = nx.read_edgelist(f, nodetype=int, data=[('capacity', int)])
                    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
                    # print(f"Max flow from {source} to {sink}: {flow_value}")
                    # run the solution
                    # compare the values
                except Exception as e:
                    print(f"Test {test_files} failed")
                    print(traceback.format_exception(*sys.exc_info()))
                    exit(1)

# flow_value, flow_dict = nx.maximum_flow(G, source, sink)

# # Output the results
# print("Generated Random Network:")
# print("Nodes:", G.nodes())
# print("Edges with capacities:")
# for (u, v, capacity) in G.edges(data='capacity'):
#     print(f"Edge ({u} -> {v}) with capacity {capacity}")

# print("\nMaximum Flow:", flow_value)
# print("Flow Distribution:", flow_dict)
