## Overview
STMatch is a stack-based subgraph pattern matching system on GPU. 

### Compile STMatch
Go to home directory of STMatch and compile the project
```Shell
make clean
#Be careful, don't use make's multi-threads execution like make -j 16. 
make
```
## Reproduce the results for all "compared with STMatch"
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/1.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/2.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/3.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/4.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/5.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/6.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/7.g
./bin/table_edge_ulb.exe ../../data/bin_graph/email-Enron/snap.txt ../../pattern/8.g

To reproduce the data sets mico, youtube, orkut change the data set path in above commands.

The speedup ratio is STMatch's matching time divided by our system's (STMatch/ours).