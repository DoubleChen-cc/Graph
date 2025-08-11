#include "src/config.h"
#include "src/graph.h"


using namespace std; 
using namespace STMatch;
int main(int argc, char* argv[]) {
    string data=argv[1];
    string gname="../data/bin_graph/"+data+"/snap.txt";
    GraphPreprocessor g(gname);
    vector<graph_edge_t> keyrow;
    vector<graph_node_t> keycol;
    keyrow.push_back(0);

    for (int i = 0; i < g.g.nnodes; i++) {
        int degree = g.g.rowptr[i + 1] - g.g.rowptr[i]; 
        if (degree > WARP_SIZE*16) {
            int step=degree/WARP_SIZE;
            keyrow.push_back(keyrow.back()+WARP_SIZE);
            graph_edge_t start = g.g.rowptr[i]; 
            for (int j = 0; j < WARP_SIZE; j++) {
                keycol.push_back(g.g.colidx[start+j*step]);
            }
        }else{
            keyrow.push_back(keyrow.back());
        }
    }

    std::ofstream keyrow_file((gname + ".keyrow.bin").c_str(), std::ios::binary);
if (keyrow_file.is_open()) {
    keyrow_file.write(reinterpret_cast<const char*>(keyrow.data()), keyrow.size() * sizeof(graph_edge_t));
    keyrow_file.close();
} else {
    std::cerr << "Failed to open keyrow binary file\n";
}

std::ofstream keycol_file((gname + ".keycol.bin").c_str(), std::ios::binary);
if (keycol_file.is_open()) {
    keycol_file.write(reinterpret_cast<const char*>(keycol.data()), keycol.size() * sizeof(graph_node_t));
    keycol_file.close();
} else {
    std::cerr << "Failed to open keycol binary file\n";
}

std::cout<<gname<<' '<<"done!"<<std::endl;
    }