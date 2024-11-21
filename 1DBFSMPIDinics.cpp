#include <iostream>
#include <vector>
#include <queue>
#include <mpi.h>
#include <chrono>
#include <cstring>
using namespace std;
using namespace std::chrono;

const int INF = 1e9;

// Structure to represent edges
struct Edge {
    int to, rev;
    int capacity, flow;
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level;
    vector<size_t> ptr;
    int n;

    Dinic(int n) : n(n) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void addEdge(int u, int v, int capacity) {
        Edge a = {v, (int)adj[v].size(), capacity, 0};
        Edge b = {u, (int)adj[u].size(), 0, 0};
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    // Parallel BFS using MPI for level graph construction
    bool parallel_bfs(int source, int sink, int rank, int size) {
        // Initialize level array
        fill(level.begin(), level.end(), -1);
        level[source] = 0;

        // Use a queue for the BFS
        queue<int> q;
        if (rank == 0) {
            q.push(source);
        }

        // Distribute BFS work across processors
        while (true) {
            vector<int> current_frontier;

            // Collect current frontier on each processor
            if (!q.empty()) {
                while (!q.empty()) {
                    current_frontier.push_back(q.front());
                    q.pop();
                }
            }

            // Share the current frontier with all processors
            int frontier_size = current_frontier.size();
            vector<int> recv_frontier;
            vector<int> recv_counts(size);
            MPI_Allgather(&frontier_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            int total_frontier_size = 0;
            for (int count : recv_counts) total_frontier_size += count;
            recv_frontier.resize(total_frontier_size);

            vector<int> displs(size, 0);
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            }

            MPI_Allgatherv(current_frontier.data(), frontier_size, MPI_INT, recv_frontier.data(),
                           recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

            // Process the received frontier
            bool updated = false;
            for (int u : recv_frontier) {
                for (const Edge& e : adj[u]) {
                    if (e.flow < e.capacity && level[e.to] == -1) {
                        level[e.to] = level[u] + 1;
                        if (rank == e.to / (n / size)) { // If vertex belongs to this processor
                            q.push(e.to);
                        }
                        updated = true;
                    }
                }
            }

            // Check if BFS is done
            int local_updated = updated ? 1 : 0;
            int global_updated;
            MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            if (!global_updated) break;
        }

        return level[sink] != -1;
    }

    // DFS for augmenting flows
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

    // Max Flow computation using the parallelized BFS
    int maxFlow(int source, int sink, int rank, int size) {
        int flow = 0;
        while (parallel_bfs(source, sink, rank, size)) {
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
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: " << argv[0] << " <input file>" << endl;
        MPI_Finalize();
        return 1;
    }

    if (freopen(argv[1], "r", stdin) == nullptr) {
        if (rank == 0) cerr << "Error: Could not open file " << argv[1] << endl;
        MPI_Finalize();
        return 1;
    }

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

    auto init_end = high_resolution_clock::now();
    double init_time = duration_cast<nanoseconds>(init_end - init_start).count();
    if (rank == 0) cout << "Initialization Time: " << init_time << " nanoseconds" << endl;

    auto comp_start = high_resolution_clock::now();

    // Compute maximum flow using parallelized BFS
    int max_flow = dinic.maxFlow(source, sink, rank, nprocs);

    auto comp_end = high_resolution_clock::now();
    double comp_time = duration_cast<nanoseconds>(comp_end - comp_start).count();

    if (rank == 0) {
        cout << "Computation Time: " << comp_time << " nanoseconds" << endl;
        cout << "Maximum Flow: " << max_flow << endl;
    }

    MPI_Finalize();
    return 0;
}
