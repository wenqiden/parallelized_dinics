#include <iostream>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <queue>
#include <atomic>

using namespace std;
using namespace std::chrono;

const int INF = 1e9;

// Structure to represent edges
struct Edge {
    int to, rev;
    int capacity, flow;
    std::shared_ptr<std::mutex> edgeLock;
    std::shared_ptr<std::atomic<int>> accessCount;

    // Constructor
    Edge(int to, int rev, int capacity, int flow, std::shared_ptr<std::mutex> edgeLock)
        : to(to), rev(rev), capacity(capacity), flow(flow), edgeLock(edgeLock), accessCount(std::make_shared<std::atomic<int>>(0)) {}
};

struct CompareEdges {
    bool operator()(Edge* a, Edge* b) {
        return a->accessCount->load() > b->accessCount->load(); // Min-heap: prioritize edges with lower access count
    }
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level;
    vector<std::unique_ptr<std::atomic<bool>>> deadEnd;
    int n;

    Dinic(int n) : n(n) {
        adj.resize(n);
        level.resize(n);
        deadEnd.resize(n);

        // Initialize all elements of deadEnd to false
        for (int i = 0; i < n; ++i) {
            deadEnd[i] = std::make_unique<std::atomic<bool>>(false);
        }
    }

    // Add an edge to the graph
    void addEdge(int u, int v, int capacity) {
        auto edgeLock = std::make_shared<std::mutex>();
        adj[u].emplace_back(Edge(v, adj[v].size(), capacity, 0, edgeLock));
        adj[v].emplace_back(Edge(u, adj[u].size() - 1, 0, 0, edgeLock));
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

    // Modified DFS function to return a boolean indicating if a path was found
    bool parallel_dfs(int u, int sink, vector<Edge*>& path) {
        if (u == sink) return true;

        // Check if this node is already marked as a dead end
        if (deadEnd[u]->load()) return false;

        bool allNeighborsDeadEnds = true;

        // Use a priority queue to sort the edges by access count
        priority_queue<Edge*, vector<Edge*>, CompareEdges> pq;

        // Add all edges from adj[u] to the priority queue
        for (Edge& edge : adj[u]) {
            pq.push(&edge);
        }

        // Extract edges from the priority queue into a sorted vector
        vector<Edge*> sortedEdges;
        while (!pq.empty()) {
            sortedEdges.push_back(pq.top());
            pq.pop();
        }

        // Traverse edges in the order of their access count
        for (Edge* edge : sortedEdges) {
            // Skip edges that do not respect the level graph or are saturated
            if (level[edge->to] != level[u] + 1 || edge->flow >= edge->capacity)
                continue;

            // Increment access count for this edge
            (*edge->accessCount)++;

            // Perform DFS on the adjacent node
            if (parallel_dfs(edge->to, sink, path)) {
                path.push_back(edge);
                return true;
            }

            // If DFS failed, decrement access count since the attempt was unsuccessful
            (*edge->accessCount)--;

            // Check if the neighbor is not a dead end
            if (!deadEnd[edge->to]->load()) {
                allNeighborsDeadEnds = false;
            }
        }

        // Mark as a dead end if all neighbors are confirmed dead ends
        if (allNeighborsDeadEnds) {
            deadEnd[u]->store(true);
        }

        return false;
    }

    // Lock and update flow once a path is found
    int lock_and_update_flow(vector<Edge*>& path) {
        int pushedFlow = INF;

        // // Try to lock each edge in the path one by one
        // for (size_t i = 0; i < path.size(); ++i) {
        //     if (!path[i]->edgeLock->try_lock()) {
        //         // If we fail to lock, release all previously acquired locks
        //         for (size_t j = 0; j < i; ++j) {
        //             path[j]->edgeLock->unlock();
        //         }
        //         // Decrement access count for all edges in the path
        //         for (Edge* edge : path) {
        //             (*edge->accessCount)--;
        //         }
        //         return 0;
        //     }
        // }

        // Spin to acquire each lock in the path one by one
        for (size_t i = 0; i < path.size(); ++i) {
            while (!path[i]->edgeLock->try_lock()) {
                // Busy wait until the lock is acquired
                // (Consider adding a small sleep or backoff for fairness if needed)
            }
        }

        // At this point, all edges are locked

        // Recompute the pushed flow based on the current remaining capacities
        for (Edge* edge : path) {
            pushedFlow = min(pushedFlow, edge->capacity - edge->flow);
        }

        // If no flow can be pushed, release all locks and decrement access counts
        if (pushedFlow == 0) {
            for (Edge* edge : path) {
                edge->edgeLock->unlock();
                (*edge->accessCount)--;
            }
            return 0;
        }

        // Update the flow along the path
        for (Edge* edge : path) {
            edge->flow += pushedFlow;
            adj[edge->to][edge->rev].flow -= pushedFlow;
        }

        // Release all locks and decrement access counts
        for (Edge* edge : path) {
            edge->edgeLock->unlock();
            (*edge->accessCount)--;
        }

        return pushedFlow;
    }

    // Max flow computation using parallel DFS and flow update
    int maxFlow(int source, int sink) {
        int totalFlow = 0;
        while (bfs(source, sink)) {
            // Reset all deadEnd flags before starting the DFS phase
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                deadEnd[i]->store(false);
            }

            #pragma omp parallel
            {
                vector<Edge*> path;
                bool foundPath;
                do {
                    path.clear();
                    foundPath = parallel_dfs(source, sink, path);
                    if (foundPath) {
                        int pushedFlow = lock_and_update_flow(path);
                        if (pushedFlow > 0) {
                            #pragma omp atomic
                            totalFlow += pushedFlow;
                        }
                    }
                } while (foundPath);
            }
        }
        return totalFlow;
    }
};

// Buffered input for faster reading
const int BUFFER_SIZE = 1 << 30; // 1 GB buffer
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

    int num_threads = atoi(argv[2]);
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