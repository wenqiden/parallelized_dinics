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
    vector<int> local_row_ptr, local_col_idx, local_capacities, local_flows;
    vector<int> level;
    vector<size_t> ptr;
    int n;
    int rank;
    int nprocs;

    Dinic(int n, int rank, int nprocs) 
        : n(n), rank(rank), nprocs(nprocs) {}

    void initialize(const vector<int>& row_ptr, const vector<int>& col_idx,
                    const vector<int>& capacities, const vector<int>& flows) {
        local_row_ptr = row_ptr;
        local_col_idx = col_idx;
        local_capacities = capacities;
        local_flows.resize(capacities.size(), 0);
        level.resize(n, -1);
        ptr.resize(n);
    }

    // void addEdge(int u, int v, int capacity) {
    //     Edge a = {v, (int)adj[v].size(), capacity, 0};
    //     Edge b = {u, (int)adj[u].size(), 0, 0};
    //     adj[u].push_back(a);
    //     adj[v].push_back(b);
    // }

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

// Function to partition vertices among processes
void partition_vertices(int n, int rank, int size, int& local_n, int& start_vertex, int& end_vertex) {
    int base_vertices = n / size;
    int remainder = n % size;

    start_vertex = rank * base_vertices + std::min(rank, remainder);
    end_vertex = start_vertex + base_vertices + (rank < remainder);
    local_n = end_vertex - start_vertex;
}

// Function to scatter CSR data to all processes
void scatter_csr(
    int n, int m, int rank, int size,
    const std::vector<int>& global_row_ptr, const std::vector<int>& global_col_idx,
    const std::vector<int>& global_capacities, const std::vector<int>& global_flows,
    std::vector<int>& local_row_ptr, std::vector<int>& local_col_idx,
    std::vector<int>& local_capacities, std::vector<int>& local_flows) {

    // Step 1: Partition vertices
    int local_n, start_vertex, end_vertex;
    partition_vertices(n, rank, size, local_n, start_vertex, end_vertex);

    // Step 2: Calculate row_ptr counts and displacements
    std::vector<int> send_row_counts(size), send_row_displs(size);
    std::vector<int> send_col_counts(size), send_col_displs(size);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int proc_start, proc_end, proc_size;
            partition_vertices(n, i, size, proc_size, proc_start, proc_end);

            // Row pointers: Each process gets `proc_size + 1` rows
            send_row_counts[i] = proc_size + 1;

            if (i > 0)
                send_row_displs[i] = send_row_displs[i-1] + send_row_counts[i-1];
            else
                send_row_displs[i] = 0;

            // Column and capacity counts: Based on row_ptr
            send_col_counts[i] = global_row_ptr[proc_end] - global_row_ptr[proc_start];
            send_col_displs[i] = global_row_ptr[proc_start];
        }
    }

    // Step 3: Scatter row_ptr
    int local_row_count = local_n + 1;
    local_row_ptr.resize(local_row_count);
    // cout << "rank " << rank << " send_row_counts: ";
    // for (int i = 0; i < send_row_counts.size(); i++) {
    //     cout << send_row_counts[i] << " ";
    // }
    // cout << endl;

    // cout << "rank " << rank << " send_row_displs: ";
    // for (int i = 0; i < send_row_displs.size(); i++) {
    //     cout << send_row_displs[i] << " ";
    // }
    // cout << endl;

    // cout << "rank " << rank << " local_row_count: " << local_row_count << endl;
    MPI_Scatterv(global_row_ptr.data(), send_row_counts.data(), send_row_displs.data(), MPI_INT,
                 local_row_ptr.data(), local_row_count, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "rank " << rank << " local_row_ptr after scatter: ";
    for (int i = 0; i < local_row_ptr.size(); i++) {
        cout << local_row_ptr[i] << " ";
    }
    cout << endl;

    // Adjust local_row_ptr to start from 0
    int offset = local_row_ptr[0];
    for (int& row : local_row_ptr) {
        row -= offset;
    }

    // Step 4: Scatter col_idx, capacities, and flows
    int local_col_count = rank == 0 ? send_col_counts[rank] : 0;
    local_col_idx.resize(local_col_count);
    local_capacities.resize(local_col_count);
    local_flows.resize(local_col_count);

    MPI_Scatterv(global_col_idx.data(), send_col_counts.data(), send_col_displs.data(), MPI_INT,
                 local_col_idx.data(), local_col_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_capacities.data(), send_col_counts.data(), send_col_displs.data(), MPI_INT,
                 local_capacities.data(), local_col_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_flows.data(), send_col_counts.data(), send_col_displs.data(), MPI_INT,
                 local_flows.data(), local_col_count, MPI_INT, 0, MPI_COMM_WORLD);
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
    if (rank == 0) {
        // Read graph data
        n = fast_read_int();
        m = fast_read_int();
        source = fast_read_int();
        sink = fast_read_int();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sink, 1, MPI_INT, 0, MPI_COMM_WORLD);


    Dinic dinic(n, rank, nprocs);

    int base_vertices = n / nprocs;
    int extra_vertices = n % nprocs;
    int local_block_size = base_vertices + (rank < extra_vertices ? 1 : 0);

    dinic.rank = rank;
    dinic.nprocs = nprocs;

    vector<int> row_ptr, col_idx, capacities, flows;

    // Step 1: Read edges and build CSR
    if (rank == 0) {
        vector<tuple<int, int, int>> edges;

        for (int i = 0; i < m; ++i) {
            int u = fast_read_int();
            int v = fast_read_int();
            int capacity = fast_read_int();
            edges.emplace_back(u, v, capacity);
        }

         row_ptr.resize(n + 1, 0);
        for (const auto& [u, v, c] : edges) {
            row_ptr[u + 1]++; // Increment outdegree for vertex u
        }

        // Step 3: Build row_ptr (prefix sum of outdegrees)
        for (int i = 1; i <= n; i++) {
            row_ptr[i] += row_ptr[i - 1];
        }


        std::cout << "rank " << rank << " row_ptr: ";
        for (int i = 0; i < row_ptr.size(); i++) {
            std::cout << row_ptr[i] << " ";
        }
        std::cout << std::endl;

        // Step 4: Populate col_idx, capacities, and flows
        col_idx.resize(m);
        capacities.resize(m);
        flows.resize(m, 0); // Initialize flows to 0
        std::vector<int> current_index = row_ptr; // Copy of row_ptr for edge insertion

        for (const auto& [u, v, c] : edges) {
            int index = current_index[u]++;
            col_idx[index] = v;
            capacities[index] = c;
            flows[index] = 0; // Initially, all flows are 0
        }
    }

    std::vector<int> local_row_ptr, local_col_idx, local_capacities, local_flows;    

    scatter_csr(n, m, rank, nprocs, row_ptr, col_idx, capacities, flows,
                local_row_ptr, local_col_idx, local_capacities, local_flows);

    std::cout << "Process " << rank << " owns vertices [" << local_row_ptr.size() - 1
              << "] with " << local_col_idx.size() << " edges." << std::endl;
    
    exit(0);


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