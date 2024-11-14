#include <iostream>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>
#include <mutex>

using namespace std;
using namespace std::chrono;

const int INF = 1e9;
int num_threads;

// Structure to represent edges
struct Edge {
    int to, rev;
    int capacity, flow;
    std::shared_ptr<std::mutex> edgeLock;

    // Constructor
    Edge(int to, int rev, int capacity, int flow)
        : to(to), rev(rev), capacity(capacity), flow(flow),
          edgeLock(std::make_shared<std::mutex>()) {}
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level;
    int n;

    Dinic(int n) : n(n) {
        adj.resize(n);
        level.resize(n);
    }

    // Add an edge to the graph
    void addEdge(int u, int v, int capacity) {
        adj[u].emplace_back(Edge(v, adj[v].size(), capacity, 0));
        adj[v].emplace_back(Edge(u, adj[u].size() - 1, 0, 0));
    }

    // BFS to build the level graph
    bool bfs(int source, int sink) {
        fill(level.begin(), level.end(), -1);
        level[source] = 0;
        queue<int> q;
        q.push(source);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (const Edge &e : adj[u]) {
                if (e.flow < e.capacity && level[e.to] == -1) {
                    level[e.to] = level[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[sink] != -1;
    }

    // Parallel DFS with locking
    int parallel_dfs(int u, int sink, int flow) {
        if (u == sink) return flow;

        for (size_t i = 0; i < adj[u].size(); ++i) {
            Edge &edge = adj[u][i];

            // Skip if the edge doesn't respect the level graph or has no capacity left
            if (level[edge.to] != level[u] + 1 || edge.flow >= edge.capacity)
                continue;

            int remainingCapacity = edge.capacity - edge.flow;
            int pushedFlow = 0;

            // Attempt to acquire the lock using try_lock()
            if (edge.edgeLock->try_lock()) {
                // Perform DFS on the adjacent node while holding the lock
                pushedFlow = parallel_dfs(edge.to, sink, min(flow, remainingCapacity));
                
                // If we successfully pushed some flow, update the edge flows
                if (pushedFlow > 0) {
                    edge.flow += pushedFlow;
                    adj[edge.to][edge.rev].flow -= pushedFlow;
                    edge.edgeLock->unlock(); // Unlock only after updating the flow
                    return pushedFlow;
                }

                // Unlock if no flow was pushed
                edge.edgeLock->unlock();
            }
        }
        return 0;
    }

    int maxFlow(int source, int sink) {
        int totalFlow = 0;
        while (bfs(source, sink)) {
            #pragma omp parallel reduction(+:totalFlow)
            {
                int pushed;
                while ((pushed = parallel_dfs(source, sink, INF)) > 0) {
                    totalFlow += pushed;
                }
            }
        }
        return totalFlow;
    }
};

// Buffered input for faster reading
const int BUFFER_SIZE = 1 << 20; // 1 MB buffer
char buffer[BUFFER_SIZE];
size_t buffer_pos = 0, buffer_len = 0;

inline char get_char() {
    if (buffer_pos == buffer_len) {
        buffer_len = fread(buffer, 1, BUFFER_SIZE, stdin);
        buffer_pos = 0;
    }
    return buffer[buffer_pos++];
}

inline int fast_read_int() {
    int x = 0;
    char c = get_char();
    while (c < '0' || c > '9') c = get_char(); // Skip non-digit characters
    while (c >= '0' && c <= '9') {
        x = x * 10 + (c - '0');
        c = get_char();
    }
    return x;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input file> <num_threads>" << endl;
        return 1;
    }

    num_threads = atoi(argv[2]);
    if (num_threads <= 0) {
        cerr << "Error: num_threads must be a positive integer" << endl;
        return 1;
    }
    omp_set_num_threads(num_threads);

    // Start measuring initialization time
    auto init_start = high_resolution_clock::now();

    // Use freopen to redirect stdin to the input file
    if (freopen(argv[1], "r", stdin) == nullptr) {
        cerr << "Error: Could not open file " << argv[1] << endl;
        return 1;
    }

    int n = fast_read_int();
    int m = fast_read_int();
    Dinic dinic(n);

    int source = fast_read_int();
    int sink = fast_read_int();

    for (int i = 0; i < m; ++i) {
        int u = fast_read_int();
        int v = fast_read_int();
        int capacity = fast_read_int();
        dinic.addEdge(u, v, capacity);
    }

    // End measuring initialization time
    auto init_end = high_resolution_clock::now();
    double init_time = duration_cast<nanoseconds>(init_end - init_start).count();
    cout << "Initialization Time: " << init_time << " nanoseconds" << endl;

    // Start measuring computation time
    auto comp_start = high_resolution_clock::now();

    // Compute the maximum flow
    int max_flow = dinic.maxFlow(source, sink);

    // End measuring computation time
    auto comp_end = high_resolution_clock::now();
    double comp_time = duration_cast<nanoseconds>(comp_end - comp_start).count();
    cout << "Computation Time: " << comp_time << " nanoseconds" << endl;

    // Output the maximum flow result
    cout << "Maximum Flow: " << max_flow << endl;

    return 0;
}