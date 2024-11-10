#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

// Structure to represent edges
struct Edge {
    int to, rev;
    int capacity, flow;
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level, ptr;
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

    bool bfs(int source, int sink) {
        fill(level.begin(), level.end(), -1);
        level[source] = 0;
        queue<int> q;
        q.push(source);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (const Edge& e : adj[u]) {
                if (e.flow < e.capacity && level[e.to] == -1) {
                    level[e.to] = level[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[sink] != -1;
    }

    int dfs(int u, int sink, int pushed) {
        if (u == sink) return pushed;
        for (int& i = ptr[u]; i < adj[u].size(); ++i) {
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
        while (bfs(source, sink)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(source, sink, INF)) {
                flow += pushed;
            }
        }
        return flow;
    }
};

int main() {
    int n, m;
    cin >> n >> m;
    Dinic dinic(n);

    for (int i = 0; i < m; ++i) {
        int u, v, capacity;
        cin >> u >> v >> capacity;
        dinic.addEdge(u, v, capacity);
    }

    int source, sink;
    cin >> source >> sink;

    cout << "Maximum Flow: " << dinic.maxFlow(source, sink) << endl;
    return 0;
}