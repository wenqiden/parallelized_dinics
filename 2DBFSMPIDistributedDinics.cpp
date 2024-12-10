#include <iostream>
#include <vector>
#include <queue>
#include <mpi.h>
#include <chrono>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <numeric>
using namespace std;
using namespace std::chrono;

const int INF = 1e9;
int P_COLUMN = 2;

// Function to partition vertices among processes
void partition_vertices(int n, int rank, int size, int& local_n, int& local_m,
                        int& start_row, int& end_row, 
                        int& start_col, int& end_col) {
    int base_vertices_col = (n + P_COLUMN - 1) / P_COLUMN;
    // int remainder_col = n % P_COLUMN;
    int base_vertices_row = (n + (size / P_COLUMN) - 1) / (size / P_COLUMN);
    // int remainder_row = n % (size / P_COLUMN);

    // Partition vertices for 2D block distribution
    start_row = (rank / P_COLUMN) * base_vertices_row;
    end_row = min(n, start_row + base_vertices_row);
    start_col = (rank % P_COLUMN) * base_vertices_col;
    end_col = min(n, start_col + base_vertices_col);
    local_n = (end_row - start_row);
    local_m = (end_col - start_col);
}

// Function to scatter CSR data to all processes
void scatter_adjacency_matrix(
    int n, int m, int rank, int size, 
    const std::vector<int>& global_block, 
    std::vector<int>& local_block, int& local_n, int& local_m) {

    int start_row, end_row, start_col, end_col;

    partition_vertices(n, rank, size, local_n, local_m, start_row, end_row, start_col, end_col);
    MPI_Datatype box_type;
    int sizes[2] = {n, n};
    int subsizes[2] = {local_n, local_m};
    int starts[2] = {start_row, start_col};
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &box_type);
    MPI_Type_commit(&box_type);

    std::vector<int> displs(size);
    std::vector<int> sendcounts(size, 1);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int proc_start_row, proc_end_row, proc_size_row;
            int proc_start_col, proc_end_col, proc_size_col;
            partition_vertices(n, i, size, proc_size_row, proc_size_col, proc_start_row, proc_end_row, proc_start_col, proc_end_col);
            displs[i] = proc_start_row * n + proc_start_col;
        }
    }

    // std::vector<int> local_block(local_n * local_m);
    local_block.resize(local_n * local_m);

    // distribute the submatrices
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            MPI_Send(&global_block[displs[i]], 1, box_type, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_m; j++) {
                local_block[i * local_m + j] = global_block[(start_row + i) * n + start_col + j];
            }
        }
    } else {
        MPI_Recv(local_block.data(), local_n * local_m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

class Dinic {
public:
    // vector<vector<Edge>> adj;
    vector<int> local_capacities, local_flows, global_capacities, global_flows; 
    vector<int> global_level;
    vector<size_t> ptr;
    int n, m;
    int local_n, local_m;
    int rank;
    int nprocs;

    Dinic(int n, int m, int rank, int nprocs) 
        : n(n), m(m), rank(rank), nprocs(nprocs) {}

    void initialize(const vector<int>& local_capacities, 
                    const vector<int>& global_capacities,
                    int local_n, int local_m) {
        this->local_capacities = local_capacities;
        this->local_flows.resize(local_capacities.size(), 0);
        if (rank == 0) {
            this->ptr.resize(n);
            this->global_level.resize(n, -1);
            this->global_capacities = global_capacities;
            this->global_flows.resize(n*n, 0);
        }
        this->local_n = local_n;
        this->local_m = local_m;
    }

    int find_owner(int from_vertex) {
        int base_vertices_col = (n + P_COLUMN - 1) / P_COLUMN;
        int base_vertices_row = (n + (nprocs / P_COLUMN) - 1) / (nprocs / P_COLUMN);
        int row = from_vertex / base_vertices_row;
        int col = from_vertex / base_vertices_col;

        row = min(row, nprocs / P_COLUMN - 1);
        col = min(col, P_COLUMN - 1);

        int processor_rank = row * P_COLUMN + col;

        return processor_rank;
    }

    int adjust_index(int vertex) {
        // find relative index of local vertex
        int block_row = (n + (nprocs / P_COLUMN) - 1) / (nprocs / P_COLUMN);
        int adjusted_vertex = vertex % block_row;
        return adjusted_vertex;
    }

    bool parallel_bfs(int source, int sink) {
        int row_index = rank / P_COLUMN;
        int col_index = rank % P_COLUMN;
        int block_rows = (n + (nprocs / P_COLUMN) - 1) / (nprocs / P_COLUMN);
        int block_cols = (n + P_COLUMN - 1) / P_COLUMN;

        vector<int> local_level(local_n);
        fill(local_level.begin(), local_level.end(), -1); // Reset levels
        // fill(level.begin(), level.end(), -1); // Reset levels

        vector<int> current_frontier;
        vector<int> next_frontier; 

        if (find_owner(source) == rank) {
            int local_source = adjust_index(source);
            local_level[local_source] = 0; // Set the source level
            current_frontier.push_back(source); 
        }

        int current_level = 0;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, rank / P_COLUMN, rank, &row_comm);
        MPI_Comm_split(MPI_COMM_WORLD, rank % P_COLUMN, rank, &col_comm);

        while (true) {
            int local_size = current_frontier.size();
            std::vector<int> all_sizes(P_COLUMN, 0); // Assuming P_cols processes per row
            MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, row_comm);

            std::vector<int> displs(P_COLUMN, 0);
            int total_frontier = all_sizes[0];
            for (int i = 1; i < P_COLUMN; ++i) {
                displs[i] = displs[i - 1] + all_sizes[i - 1];
                total_frontier += all_sizes[i];
            }
            std::vector<int> send_buffer = current_frontier;
            std::vector<int> recv_buffer(total_frontier, -1);
            MPI_Allgatherv(send_buffer.data(), local_size, MPI_INT,
                      recv_buffer.data(), all_sizes.data(), displs.data(), MPI_INT,
                      row_comm);

            std::unordered_map<int, std::vector<int>> neighbors_by_owner;
            for (int v : recv_buffer) {
                if (v == -1 || v >= n) continue; // Skip invalid entries

                // Determine the owner of vertex v
                int v_row_owner, v_col_owner;
                int v_owner = find_owner(v);

                // Each process scans its local adjacency matrix for all vertices in the frontier
                // to discover neighbors in its own column block
                for (int local_row = 0; local_row < block_rows; ++local_row) {
                    int global_vertex = row_index * block_rows + local_row;
                    if (global_vertex != v) continue; // Only process the current vertex

                    for (int u = 0; u < block_cols; ++u) {
                        if (local_capacities[local_row * block_cols + u] - local_flows[local_row * block_cols + u] > 0) { // Edge exists
                            int global_u = col_index * block_cols + u;

                            if (global_u >= n) continue; // Skip out-of-bounds

                            // Determine the owner of vertex u
                            int u_row_owner, u_col_owner;
                            int u_owner = find_owner(global_u);

                            // Add u to the list for its owning processor
                            neighbors_by_owner[u_owner].push_back(global_u);
                        }
                    }
                }
            }
 
            // Prepare send counts and send buffers
            int num_row = nprocs / P_COLUMN;
            std::vector<int> send_counts(nprocs, 0);
            std::vector<int> send_counts_col(num_row, 0);
            // std::vector<int> send_displs_buffer(nprocs, 0);
            std::vector<int> send_displs_col(num_row, 0);
            std::vector<int> send_buffer_neighbors;

            // for (auto &[owner, neighbors] : neighbors_by_owner) {
            for (int owner = 0; owner < nprocs; ++owner) {
                // cout << "rank " << rank << " current owner: " << owner << " neighbors: ";
                // for (int u : neighbors) {
                //     cout << u << " ";
                // }
                // cout << endl;
                vector<int> neighbors = neighbors_by_owner[owner];
                if (neighbors.empty()) continue;
                send_counts[owner] = neighbors.size();
                send_counts_col[owner / P_COLUMN] = neighbors.size();
                send_buffer_neighbors.insert(send_buffer_neighbors.end(), neighbors.begin(), neighbors.end());
            }

            // Calculate send displacements
            int send_total = 0;
            for (int i = 0; i < num_row; ++i) {
                // send_displs_buffer[i] = send_total;
                // send_total += send_counts[i];
                send_displs_col[i] = send_total;
                send_total += send_counts_col[i];
            }

            // Exchange send counts to determine receive counts
            std::vector<int> recv_counts_buffer(nprocs, 0);

            MPI_Alltoall(send_counts.data(), P_COLUMN, MPI_INT,
                        recv_counts_buffer.data(), P_COLUMN, MPI_INT, col_comm);

            // Calculate receive displacements
            std::vector<int> recv_counts_col(num_row, 0);
            std::vector<int> recv_displs_col(num_row, 0);

            for (int i = 0; i < num_row; ++i) {
                recv_counts_col[i] += recv_counts_buffer[i * P_COLUMN + rank % P_COLUMN];
            }
            int recv_total = recv_counts_col[0];
            for (int i = 1; i < num_row; ++i) {
                recv_displs_col[i] = recv_displs_col[i - 1] + recv_counts_col[i - 1];
                recv_total += recv_counts_col[i];
            }

            // Prepare receive buffer
            std::vector<int> recv_buffer_neighbors(recv_total, -1);
            // Perform MPI_Alltoallv to send and receive neighbors
            MPI_Alltoallv(send_buffer_neighbors.data(), send_counts_col.data(), send_displs_col.data(), MPI_INT,
                        recv_buffer_neighbors.data(), recv_counts_col.data(), recv_displs_col.data(), MPI_INT,
                        col_comm);

            std::vector<int> next_frontier;

            for (int u : recv_buffer_neighbors) {
                if (u == -1 || u >= n) continue; // Skip invalid entries

                // Since communication ensures u is owned by this process, directly update
                int local_u = u % block_rows;

                // Check if u has not been visited
                if (local_level[local_u] == -1) {
                    local_level[local_u] = current_level + 1;
                    next_frontier.push_back(u);
                }
            }

            // Step 5: Check if any process still has a non-empty frontier
            int local_has_frontier = !next_frontier.empty() ? 1 : 0;
            int global_has_frontier = 0;
            MPI_Allreduce(&local_has_frontier, &global_has_frontier, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

            if (global_has_frontier == 0) {
                break;
            }

            // Update the local frontier for the next iteration
            current_frontier = next_frontier;
            current_level++;
        }

        vector<int> reduced_level(local_n);

        MPI_Allreduce(local_level.data(), reduced_level.data(), local_n, MPI_INT, MPI_MAX, row_comm);

        // all gather local levels to global level
        if (rank % P_COLUMN == 0) {
            MPI_Gather(reduced_level.data(), local_n, MPI_INT,
                        global_level.data(), local_n, MPI_INT, 0, col_comm);
        }

        

        bool reachable;

        if (rank == 0) {
            reachable = global_level[sink] != -1;
        }
        MPI_Bcast(&reachable, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Check if the sink is reachable
        return reachable;
    }

    // DFS for augmenting flows
    int dfs(int u, int sink, int pushed) {
        if (u == sink) return pushed;
        int start_idx = u * n;
        int end_idx = (u + 1) * n;
        for (size_t& v = ptr[u]; v < n; ++v) {
            int idx = start_idx + v;
            if (global_flows[idx] < global_capacities[idx] && global_level[v] == global_level[u] + 1) {
                int tr = dfs(v, sink, min(pushed, global_capacities[idx] - global_flows[idx]));
                if (tr > 0) {
                    global_flows[idx] += tr;
                    int rev_idx = v * n + u;
                    global_flows[rev_idx] -= tr;
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
            if (rank == 0) {
                fill(ptr.begin(), ptr.end(), 0);
                while (int pushed = dfs(source, sink, INF)) {
                    flow += pushed;
                }
            }
            // MPI_Barrier(MPI_COMM_WORLD);
            // MPI_Scatterv(global_flows.data(), send_col_counts.data(), send_col_displs.data(), MPI_INT,
            //      local_flows.data(), local_col_count, MPI_INT, 0, MPI_COMM_WORLD);
            scatter_adjacency_matrix(n, m, rank, nprocs, global_flows, local_flows, local_n, local_m);
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

int round_n(int n, int nprocs) {
    // find lcm or p_row and p_col
    int lcm = (P_COLUMN / std::gcd(nprocs / P_COLUMN, P_COLUMN)) * nprocs / P_COLUMN;
    return (n + lcm - 1) / lcm * lcm;
}

int roundToPowerOf2(int num) {
    //return num == 0 ? 1 : 1 << (32 - __builtin_clz(num - 1) - 1);
    if (num == 0) return 1;
    if (num && !(num & (num - 1))) return num;
    return 1 << (32 - __builtin_clz(num));
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

    P_COLUMN = roundToPowerOf2(sqrt(nprocs));

    int n, m, source, sink;
    if (rank == 0) {
        // Read graph data
        n = fast_read_int();
        m = fast_read_int();
        source = fast_read_int();
        sink = fast_read_int();
        n = round_n(n, nprocs);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sink, 1, MPI_INT, 0, MPI_COMM_WORLD);


    Dinic dinic(n, m, rank, nprocs);

    dinic.rank = rank;
    dinic.nprocs = nprocs;

    vector<int> global_capacities(n*n, 0);

    // Step 1: Read edges and build CSR
    if (rank == 0) {
        vector<tuple<int, int, int>> edges;

        for (int i = 0; i < m; ++i) {
            int u = fast_read_int();
            int v = fast_read_int();
            int capacity = fast_read_int();
            global_capacities[u*n + v] = capacity;
        }
    }

    std::vector<int> local_capacities;
    int local_n, local_m;

    scatter_adjacency_matrix(n, m, rank, nprocs, global_capacities, local_capacities, local_n, local_m);

    // cout << "Rank: " << rank << " received block" << endl;
    // for (int i = 0; i < local_n; i++) {
    //     for (int j = 0; j < local_m; j++) {
    //         cout << local_capacities[i * local_m + j] << " ";
    //     }
    //     cout << endl;
    // }

    dinic.initialize(
        local_capacities, global_capacities, local_n, local_m
    );

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