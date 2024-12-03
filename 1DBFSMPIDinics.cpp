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
    int rank;
    int nprocs;

    Dinic(int n) : n(n) {
        adj.resize(n);
        level.resize(n, -1);
        ptr.resize(n);
    }

    void addEdge(int u, int v, int capacity) {
        Edge a = {v, (int)adj[v].size(), capacity, 0};
        Edge b = {u, (int)adj[u].size(), 0, 0};
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    int find_owner(int vertex) {
        // block distribution, allow first few processors to have one extra vertex
        int base_vertices = n / nprocs;
        int extra_vertices = n % nprocs;

        if (vertex < (base_vertices + 1) * extra_vertices) {
            return vertex / (base_vertices + 1);
        } else {
            return (vertex - extra_vertices * (base_vertices + 1)) / base_vertices + extra_vertices;
        }
    }

    int adjust_index(int vertex) {
        // find relative index of local vertex
        int base_vertices = n / nprocs;
        int extra_vertices = n % nprocs;
        int start_index = rank * base_vertices + min(rank, extra_vertices);
        return vertex - start_index;      
    }

    bool parallel_bfs(int source, int sink) {
        int base_vertices = n / nprocs;
        int extra_vertices = n % nprocs;
        int local_level_size = base_vertices + (rank < extra_vertices ? 1 : 0);
        vector<int> local_level(local_level_size);
        fill(level.begin(), level.end(), -1); // Reset levels
        fill(local_level.begin(), local_level.end(), -1); // Reset levels

        if (find_owner(source) == rank) {
            int local_source = adjust_index(source);
            local_level[local_source] = 0; // Set the source level
        }

        // local_level[source] = 0;

        vector<int> current_frontier; // Current BFS frontier
        vector<int> next_frontier;    // Next BFS frontier (local)
        if (rank == 0) current_frontier.push_back(source); // Start BFS from the source on rank 0

        int current_level = 0;

        while (true) {
            vector<vector<int>> sendBuffer(nprocs); // Buffer for vertices to send
            // ensure send buffer is empty
            for (int i = 0; i < nprocs; ++i) {
                sendBuffer[i].clear();
            }

            // Process the current frontier and populate sendBuffer
            for (int u : current_frontier) {
                for (const Edge& e : adj[u]) {
                    if (e.flow < e.capacity) { // Check if the edge can be traversed
                        int owner = find_owner(e.to);
                        // only push to buffer if not already in the buffer
                        if (find(sendBuffer[owner].begin(), sendBuffer[owner].end(), e.to) == sendBuffer[owner].end()) {
                            sendBuffer[owner].push_back(e.to);
                        }
                    }
                }
            }

            // Flatten sendBuffer for MPI_Alltoallv
            vector<int> sendData;
            vector<int> sendCounts(nprocs, 0);
            vector<int> sendDisplacements(nprocs, 0);
            for (int i = 0; i < nprocs; ++i) {
                sendCounts[i] = sendBuffer[i].size();
                sendDisplacements[i] = sendData.size();
                sendData.insert(sendData.end(), sendBuffer[i].begin(), sendBuffer[i].end());
            }

            // for (int i = 0; i < nprocs; ++i) {
            //     if (rank == i) {
            //         cout << "Rank " << rank << " sendCounts: ";
            //         for (int c : sendCounts) cout << c << " ";
            //         cout << endl;
            //     }
            // }

            // Exchange counts using MPI_Alltoall
            vector<int> recvCounts(nprocs, 0);
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);



            // Compute displacements for receiving data and allocate recvData
            vector<int> recvDisplacements(nprocs, 0);
            int totalRecv = 0;
            for (int i = 0; i < nprocs; ++i) {
                recvDisplacements[i] = totalRecv;
                totalRecv += recvCounts[i];
            }
            vector<int> recvData(totalRecv);

            // Perform the data exchange using MPI_Alltoallv
            MPI_Alltoallv(
                sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_INT,
                recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_INT, MPI_COMM_WORLD
            );

            // Process the received data and update levels
            bool updated = false;
            for (int i = 0; i < nprocs; ++i) {
                for (int j = recvDisplacements[i]; j < recvDisplacements[i] + recvCounts[i]; ++j) {
                    int u = recvData[j];
                    int adjusted_u = adjust_index(u);
                    if (local_level[adjusted_u] == -1) {
                        local_level[adjusted_u] = current_level + 1; // Set the level
                        next_frontier.push_back(u);   // Add to local next frontier
                        updated = true;
                    }
                }
            }

            // Synchronize updates across all processes
            int local_updated = updated ? 1 : 0;
            int global_updated;
            MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            // If no updates were made, BFS is complete
            if (!global_updated) break;

            // Move to the next level
            current_frontier = move(next_frontier); // Transfer ownership of the next frontier
            next_frontier.clear();
            current_level++;
        }

        // all gather local levels to global level
        vector<int> global_level(n);
        vector<int> recvCounts(nprocs);
        vector<int> displs(nprocs, 0);
        for (int i = 0; i < nprocs; ++i) {
            recvCounts[i] = (i < extra_vertices ? base_vertices + 1 : base_vertices);
            if (i > 0) displs[i] = displs[i - 1] + recvCounts[i - 1];
        }

        MPI_Barrier(MPI_COMM_WORLD);


        MPI_Allgatherv(
            local_level.data(), local_level_size, MPI_INT,
            level.data(), recvCounts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD
        );

        // Check if the sink is reachable
        return level[sink] != -1;
    }

    // Parallel BFS using MPI for level graph construction
    // bool parallel_bfs(int source, int sink) {
    //     fill(level.begin(), level.end(), -1);
    //     level[source] = 0;

    //     queue<int> q;
    //     if (rank == 0) q.push(source);

    //     while (true) {
    //         vector<vector<int>> sendBuffer(nprocs);  // Buffer for vertices to send
    //         while (!q.empty()) {
    //             int u = q.front(); q.pop();
    //             for (const Edge& e : adj[u]) {
    //                 if (e.flow < e.capacity && level[e.to] == -1) {
    //                     int owner = find_owner(e.to);
    //                     sendBuffer[owner].push_back(e.to);
    //                 }
    //             }
    //         }

    //         vector<vector<int>> recvBuffer(nprocs);
    //         vector<MPI_Request> requests(nprocs);

    //         for (int i = 0; i < nprocs; ++i) {
    //             if (!sendBuffer[i].empty()) {
    //                 if (rank != i) {
    //                     MPI_Isend(sendBuffer[i].data(), sendBuffer[i].size(), MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);
    //                 } else {
    //                     for (int u : sendBuffer[i]) {
    //                         recvBuffer[i].push_back(u);
    //                     }
    //                 }
    //             }
    //         }

    //         for (int i = 0; i < nprocs; ++i) {
    //             if (rank != i) {
    //                 int recv_size;
    //                 MPI_Status status;
    //                 MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
    //                 MPI_Get_count(&status, MPI_INT, &recv_size);
    //                 recvBuffer[i].resize(recv_size);
    //                 MPI_Recv(recvBuffer[i].data(), recv_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             }
    //         }

    //         MPI_Waitall(nprocs - 1, requests.data(), MPI_STATUSES_IGNORE);

    //         bool updated = false;
    //         for (int i = 0; i < nprocs; ++i) {
    //             for (int u : recvBuffer[i]) {
    //                 if (level[u] == -1) {
    //                     level[u] = level[find_owner(u)] + 1;
    //                     if (rank == find_owner(u)) q.push(u);
    //                     updated = true;
    //                 }
    //             }
    //         }

    //         // Synchronize levels across all processes using MPI_Allreduce
    //         int local_updated = updated ? 1 : 0;
    //         int global_updated;
    //         MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    //         if (!global_updated) break;
    //     }

    //     return level[sink] != -1;
    // }

    // bool parallel_bfs(int source, int sink) {
    //     // Initialize level array
    //     fill(level.begin(), level.end(), -1);
    //     level[source] = 0;

    //     // Use a queue for the BFS
    //     queue<int> q;
    //     if (rank == 0) {
    //         q.push(source);
    //     }

    //     // Distribute BFS work across processors
    //     while (true) {
    //         vector<int> current_frontier;

    //         // Collect current frontier on each processor
    //         if (!q.empty()) {
    //             while (!q.empty()) {
    //                 current_frontier.push_back(q.front());
    //                 q.pop();
    //             }
    //         }

    //         // Share the current frontier with all processors
    //         int frontier_size = current_frontier.size();
    //         vector<int> recv_frontier;
    //         vector<int> recv_counts(nprocs);
    //         MPI_Allgather(&frontier_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    //         int total_frontier_size = 0;
    //         for (int count : recv_counts) total_frontier_size += count;
    //         recv_frontier.resize(total_frontier_size);

    //         vector<int> displs(nprocs, 0);
    //         for (int i = 1; i < nprocs; ++i) {
    //             displs[i] = displs[i - 1] + recv_counts[i - 1];
    //         }

    //         MPI_Allgatherv(current_frontier.data(), frontier_size, MPI_INT, recv_frontier.data(),
    //                        recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    //         // Process the received frontier
    //         bool updated = false;
    //         for (int u : recv_frontier) {
    //             for (const Edge& e : adj[u]) {
    //                 if (e.flow < e.capacity && level[e.to] == -1) {
    //                     level[e.to] = level[u] + 1;
    //                     if (rank == e.to / (n / nprocs)) { // If vertex belongs to this processor
    //                         q.push(e.to);
    //                     }
    //                     updated = true;
    //                 }
    //             }
    //         }

    //         // Check if BFS is done
    //         int local_updated = updated ? 1 : 0;
    //         int global_updated;
    //         MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    //         if (!global_updated) break;
    //     }

    //     for (int i = 0; i < nprocs; ++i) {
    //         if (rank == i) {
    //             cout << "Rank " << rank << " level: ";
    //             for (int l : level) cout << l << " ";
    //             cout << endl;
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);
    //     }

    //     return level[sink] != -1;
    // }

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
    int maxFlow(int source, int sink) {
        int flow = 0;
        while (parallel_bfs(source, sink)) {
            // if (rank == 0) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(source, sink, INF)) {
                flow += pushed;
            }
            // }
            MPI_Barrier(MPI_COMM_WORLD);
            // broadcast flow to all processors
        }
        return flow;
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

    auto init_start = high_resolution_clock::now();

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

    int n, m, source, sink;
    // Read graph data
    n = fast_read_int();
    m = fast_read_int();
    source = fast_read_int();
    sink = fast_read_int();

    Dinic dinic(n);
    dinic.rank = rank;
    dinic.nprocs = nprocs;

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
    int max_flow = dinic.maxFlow(source, sink);

    auto comp_end = high_resolution_clock::now();
    double comp_time = duration_cast<nanoseconds>(comp_end - comp_start).count();

    if (rank == 0) {
        cout << "Computation Time: " << comp_time << " nanoseconds" << endl;
        cout << "Maximum Flow: " << max_flow << endl;
    }

    MPI_Finalize();
    return 0;
}
