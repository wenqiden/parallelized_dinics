import subprocess, time, filecmp, os
import sys, traceback
import networkx as nx

LARGE_TEST_DIR = "./largetest_networkx"
SOL_PATH = "../SequentialDinics.cpp"

# compile the solution
if os.path.exists("sol"):
    subprocess.run(["rm", "sol"])
subprocess.run(["g++", SOL_PATH, "-o", "sol"])

for file_name in os.listdir(LARGE_TEST_DIR):
    file_path = os.path.join(LARGE_TEST_DIR, file_name)
    if os.path.isdir(file_path):
        print("="*20)
        print(f"Testing {file_name}...")
        for test_files in os.listdir(file_path):
            test_file_path = os.path.join(file_path, test_files)
            if os.path.isfile(test_file_path):
                try:
                    # load the graph
                    with open(test_file_path, 'r') as f:
                        num_nodes, num_edges = map(int, f.readline().split())
                        source, sink = map(int, f.readline().split())
                        G = nx.read_edgelist(f, nodetype=int, data=[('capacity', int)], create_using=nx.DiGraph)
                    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
                    # run the solution
                    result = subprocess.run(["./sol", test_file_path], capture_output=True)
                    result_max_flow = result.stdout.decode().strip().split("\n")[-1].split(": ")[-1]
                    # compare the values
                    if result.returncode != 0:
                        print(f"Test {test_files} failed")
                        print("Return code is not 0")
                        print(result.stderr.decode())
                        exit(1)
                    elif result_max_flow != str(flow_value):
                        print(f"Test {test_files} failed")
                        print(f"Expected: {flow_value}")
                        print(f"Got: {result.stdout.decode()}")
                        print("Graph:", G.edges(data=True))
                        exit(1)
                    else:
                        print(f"Passed!")
                except Exception as e:
                    print(f"Test {test_files} failed")
                    print(traceback.format_exception(*sys.exc_info()))
                    exit(1)
