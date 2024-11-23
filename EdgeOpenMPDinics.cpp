#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;

#pragma omp declare reduction(vec_merge: std::vector<int>: \
    std::move(omp_out).insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = decltype(omp_orig)())

const int INF = 1e9;

class Dinic {
public:
    vector<vector<int>> capacity;  // Adjacency matrix for capacities
    vector<vector<int>> flow;      // Adjacency matrix for flows
    vector<int> level;             // Level graph
    vector<int> current_level;     // For BFS levels
    vector<int> next_level;        // For BFS levels in the next step
    vector<int> ptr;               // Current pointer for DFS
    int n;

    Dinic(int n) : n(n) {
        capacity.assign(n, vector<int>(n, 0));
        flow.assign(n, vector<int>(n, 0));
        level.resize(n);
        ptr.resize(n);
    }

    void addEdge(int u, int v, int cap) {
        capacity[u][v] += cap;  // Add capacity to adjacency matrix
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
            #pragma omp parallel for schedule(dynamic, 100000) reduction(vec_merge: next_level)
            for (size_t idx = 0; idx < current_level.size() * n; ++idx) {
                int u = current_level[idx/n];
                int v = idx % n;

                // Check if we can use the edge u -> v
                if (capacity[u][v] > flow[u][v] && level[v] == -1) {
                    // Atomically set level[v] to level[u] + 1 if it's still -1
                    if (__sync_bool_compare_and_swap(&level[v], -1, level[u] + 1)) {
                        next_level.push_back(v); // Reduction merges into next_level
                    }
                }
            }

            // Move to the next level
            current_level.swap(next_level);
        }

        return level[sink] != -1;
    }

    int dfs(int u, int sink, int pushed) {
        if (u == sink) return pushed;
        for (int& v = ptr[u]; v < n; ++v) {
            if (level[v] == level[u] + 1 && flow[u][v] < capacity[u][v]) {
                int tr = dfs(v, sink, min(pushed, capacity[u][v] - flow[u][v]));
                if (tr > 0) {
                    flow[u][v] += tr;
                    flow[v][u] -= tr;
                    return tr;
                }
            }
        }
        return 0;
    }

    int maxFlow(int source, int sink) {
        int total_flow = 0;
        while (parallel_bfs(source, sink)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(source, sink, INF)) {
                total_flow += pushed;
            }
        }
        return total_flow;
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