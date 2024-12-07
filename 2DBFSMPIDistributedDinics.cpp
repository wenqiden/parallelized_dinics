#include <iostream>
#include <vector>
#include <queue>
#include <mpi.h>
#include <chrono>
#include <cstring>
#include <cmath>
using namespace std;
using namespace std::chrono;

const int INF = 1e9;
int P_COLUNM = 2;

class Dinic {
public:
    // vector<vector<Edge>> adj;
    vector<int> local_capacities, local_flows, global_capacities, global_flows; 
    vector<int> global_level;
    vector<size_t> ptr;
    int n, m;
    // int local_n, local_m;
    int rank;
    int nprocs;

    Dinic(int n, int m, int rank, int nprocs) 
        : n(n), m(m), rank(rank), nprocs(nprocs) {}

    void initialize(const vector<int>& local_capacities, 
                    const vector<int>& global_capacities) {
        this->local_capacities = local_capacities;
        this->local_flows.resize(local_capacities.size(), 0);
        if (rank == 0) {
            this->ptr.resize(n);
            this->global_level.resize(n, -1);
            this->global_capacities = global_capacities;
            this->global_flows.resize(n*n, 0);
        }
    }

    int find_owner(int from_vertex) {
        int base_vertices_col = n / P_COLUNM;
        int remainder_col = n % P_COLUNM;
        int base_vertices_row = n / (nprocs / P_COLUNM);
        int remainder_row = n % (nprocs / P_COLUNM);

        int row_processor = from_vertex / (base_vertices_row + (from_vertex < remainder_row ? 1 : 0));

        int processor_rank = row_processor * P_COLUNM;
        return processor_rank;
    }

    int adjust_index(int vertex) {
        // find relative index of local vertex
        int base_vertices = n / nprocs;
        int extra_vertices = n % nprocs;
        int start_index = rank * base_vertices + min(rank, extra_vertices);
        return vertex - start_index;      
    }

    bool parallel_bfs(int source, int sink) {
        vector<int> local_level(local_n);
        fill(local_level.begin(), local_level.end(), -1); // Reset levels
        // fill(level.begin(), level.end(), -1); // Reset levels

        int base_vertices = n / nprocs;
        int extra_vertices = n % nprocs;

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
                int adjusted_u = adjust_index(u);
                for (int i = local_row_ptr[adjusted_u]; i < local_row_ptr[adjusted_u + 1]; ++i) {
                    int v = local_col_idx[i];
                    
                    if (local_flows[i] < local_capacities[i]) { // Check if the edge can be traversed
                        int owner = find_owner(v);
                        // only push to buffer if not already in the buffer
                        if (find(sendBuffer[owner].begin(), sendBuffer[owner].end(), v) == sendBuffer[owner].end()) {
                            sendBuffer[owner].push_back(v);
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
        vector<int> recvCounts(nprocs);
        vector<int> displs(nprocs, 0);
        for (int i = 0; i < nprocs; ++i) {
            recvCounts[i] = (i < extra_vertices ? base_vertices + 1 : base_vertices);
            if (i > 0) displs[i] = displs[i - 1] + recvCounts[i - 1];
        }

        MPI_Barrier(MPI_COMM_WORLD);


        MPI_Gatherv(
            local_level.data(), local_n, MPI_INT,
            global_level.data(), recvCounts.data(), displs.data(), MPI_INT, 0, 
            MPI_COMM_WORLD
        );

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
        int start_idx = global_row_ptr[u];
        int end_idx = global_row_ptr[u + 1];
        for (size_t& i = ptr[u]; i < end_idx - start_idx; ++i) {
            int idx = start_idx + i;
            int v = global_col_idx[idx];
            if (global_flows[idx] < global_capacities[idx] && global_level[v] == global_level[u] + 1) {
                int tr = dfs(v, sink, min(pushed, global_capacities[idx] - global_flows[idx]));
                if (tr > 0) {
                    global_flows[idx] += tr;
                    auto it = std::find(global_col_idx.begin() + global_row_ptr[v], global_col_idx.begin() + global_row_ptr[v + 1], u);
                    // cout << "it == end" << (it == global_col_idx.begin() + global_row_ptr[v + 1]) << endl;
                    // int rev_idx = global_row_ptr[v] + (it - global_col_idx.begin());
                    int rev_idx = it - global_col_idx.begin();
                    global_flows[rev_idx] -= tr;
                    // cout << "rev flow: " << global_flows[rev_idx] << endl;
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
            MPI_Scatterv(global_flows.data(), send_col_counts.data(), send_col_displs.data(), MPI_INT,
                 local_flows.data(), local_col_count, MPI_INT, 0, MPI_COMM_WORLD);
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
void partition_vertices(int n, int rank, int size, int& local_n, int& local_m,
                        int& start_row, int& end_row, 
                        int& start_col, int& end_col) {
    int base_vertices_col = n / P_COLUNM;
    int remainder_col = n % P_COLUNM;
    int base_vertices_row = n / (size / P_COLUNM);
    int remainder_row = n % (size / P_COLUNM);

    // Partition vertices for 2D block distribution
    start_row = (rank / P_COLUNM) * base_vertices_row + min(rank / P_COLUNM, remainder_row);
    end_row = start_row + base_vertices_row + (rank / P_COLUNM < remainder_row);
    start_col = (rank % P_COLUNM) * base_vertices_col + min(rank % P_COLUNM, remainder_col);
    end_col = start_col + base_vertices_col + (rank % P_COLUNM < remainder_col);
    local_n = (end_row - start_row);
    local_m = (end_col - start_col);
}

// Function to scatter CSR data to all processes
void scatter_adjacency_matrix(
    int n, int m, int rank, int size, 
    const std::vector<int>& global_block, 
    std::vector<int>& local_block) {

    P_COLUNM = min(P_COLUNM, size);

    int local_n, local_m, start_row, end_row, start_col, end_col;

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

        // cout << "displs: ";
        // for (int i = 0; i < size; i++) {
        //     cout << displs[i] << " ";
        // }
        // cout << endl;
    }

    // std::vector<int> local_block(local_n * local_m);
    local_block.resize(local_n * local_m);

    // if (rank == 0) {
    //     cout << "global_capacities: \n";
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             cout << global_capacities[i * n + j] << " ";
    //         }
    //         cout << "\n";
    //     }
    //     cout << endl;
    // }

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
    // Print the local block
    // std::cout << "Process " << rank << " received block:\n";
    // for (int i = 0; i < local_n; ++i) {
    //     for (int j = 0; j < local_m; ++j) {
    //         std::cout << local_block[i * local_m + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
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

    scatter_adjacency_matrix(n, m, rank, nprocs, global_capacities, local_capacities);

    std::cout << "Process " << rank << " received block:\n";
    for (int i = 0; i < local_capacities.size(); ++i) {
        std::cout << local_block[i] << " ";
    }
    cout << endl;

    dinic.initialize(
        local_capacities, global_capacities
    );

    auto init_end = high_resolution_clock::now();
    // double init_time = duration_cast<nanoseconds>(init_end - init_start).count();
    // if (rank == 0) cout << "Initialization Time: " << init_time << " nanoseconds" << endl;

    // auto comp_start = high_resolution_clock::now();

    // // Compute maximum flow using parallelized BFS
    // int max_flow = dinic.maxFlow(source, sink);

    // auto comp_end = high_resolution_clock::now();
    // double comp_time = duration_cast<nanoseconds>(comp_end - comp_start).count();

    // if (rank == 0) {
    //     cout << "Computation Time: " << comp_time << " nanoseconds" << endl;
    //     cout << "Maximum Flow: " << max_flow << endl;
    // }

    MPI_Finalize();
    return 0;
}