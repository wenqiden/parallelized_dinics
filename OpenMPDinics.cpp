#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;

#pragma omp declare reduction(vec_merge: std::vector<int>: \
    std::move(omp_out).insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = decltype(omp_orig)())

const int INF = 1e9;

// Structure to represent edges
struct Edge {
    int to, rev;
    int capacity, flow;
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level, current_level, next_level;
    vector<size_t> ptr;
    int n;

    Dinic(int n) : n(n) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
        current_level.reserve(n);
        next_level.reserve(n);
    }

    void addEdge(int u, int v, int capacity) {
        Edge a = {v, (int)adj[v].size(), capacity, 0};
        Edge b = {u, (int)adj[u].size(), 0, 0};
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    bool parallel_bfs(int source, int sink) {
        fill(level.begin(), level.end(), -1);
        level[source] = 0;

        current_level.clear();
        current_level.push_back(source);

        while (!current_level.empty()) {
            // Prepare for the next level
            next_level.clear();

            // Process edges in parallel
            #pragma omp parallel for schedule(dynamic) reduction(vec_merge: next_level)
            for (size_t i = 0; i < current_level.size(); ++i) {
                int u = current_level[i];
                for (size_t j = 0; j < adj[u].size(); ++j) {
                    const Edge& e = adj[u][j];
                    // Check if we can use the edge
                    if (e.flow < e.capacity && level[e.to] == -1) {
                        // Atomically set level[e.to] to level[u] + 1 if it's still -1
                        if (__sync_bool_compare_and_swap(&level[e.to], -1, level[u] + 1)) {
                            next_level.push_back(e.to);
                        }
                    }
                }
            }

            // Move to the next level
            current_level.swap(next_level);
        }

        // while (!current_level.empty()) {
        //    #pragma omp parallel
        //     {
        //         #pragma omp single
        //         {
        //             for (size_t i = 0; i < current_level.size(); ++i) {
        //                 #pragma omp task firstprivate(i)
        //                 {
        //                     vector<int> local_next_level;
        //                     int u = current_level[i];
        //                     for (size_t j = 0; j < adj[u].size(); ++j) {
        //                         const Edge& e = adj[u][j];
        //                         if (e.flow < e.capacity && level[e.to] == -1) {
        //                             if (__sync_bool_compare_and_swap(&level[e.to], -1, level[u] + 1)) {
        //                                 local_next_level.push_back(e.to);
        //                             }
        //                         }
        //                     }
        //                     #pragma omp critical
        //                     next_level.insert(next_level.end(), local_next_level.begin(), local_next_level.end());
        //                 }
        //             }
        //         }
        //     }

        //     current_level.swap(next_level);
        //     next_level.clear();
        // }

        return level[sink] != -1;
    }

    int dfs(int u, int sink, int pushed) {
        if (u == sink) return pushed;
        for (size_t& i = ptr[u]; i < adj[u].size(); ++i) {
            Edge& e = adj[u][i];
            if (e.flow < e.capacity && level[e.to] == level[u] + 1) {
                int tr = dfs(e.to, sink, min(pushed, e.capacity - e.flow));
                if (tr > 0) {
                    e.flow += tr;
                    adj[e.to][e.rev].flow -= tr;
                    return tr;
                }
            }
        }
        return 0;
    }

    int maxFlow(int source, int sink) {
        int flow = 0;
        while (parallel_bfs(source, sink)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(source, sink, INF)) {
                flow += pushed;
            }
        }
        return flow;
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