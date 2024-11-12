import subprocess, time, filecmp, os
import sys, traceback
import networkx as nx

SMALL_TEST_DIR = "./smalltest_networkx"
# SOL_PATH = "../SequentialDinics.cpp"
SOL_PATH = "../OpenMPDinics.cpp"

# compile the solution
if os.path.exists("sol"):
    subprocess.run(["rm", "sol"])

subprocess.run(["g++", SOL_PATH, "-Wall", "-O3", "-std=c++17", "-m64", "-I.", "-fopenmp", "-Wno-unknown-pragmas", "-o", "sol"])

for file_name in os.listdir(SMALL_TEST_DIR):
    file_path = os.path.join(SMALL_TEST_DIR, file_name)
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
                    if SOL_PATH == "../OpenMPDinics.cpp" or SOL_PATH == "../MPIDinics.cpp":
                        result = subprocess.run(["./sol", test_file_path, "8"], capture_output=True)
                    else:
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
                        print(f"Got: {result_max_flow}")
                        print("Graph:", G.edges(data=True))
                        exit(1)
                    else:
                        print(f"Passed!")
                except Exception as e:
                    print(f"Test {test_files} failed")
                    print(traceback.format_exception(*sys.exc_info()))
                    exit(1)

if os.path.exists("sol"):
    subprocess.run(["rm", "sol"])
print("="*20)
print("All tests passed!")
print("="*20)
