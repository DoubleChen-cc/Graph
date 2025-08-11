# AE cuTS: Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using Trie Based Data Structure

This repository contains the code for the "cuTS: Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using Trie Based Data Structure" framework. The cuTS framework is an efficient subgraph isomorphism solver for GPUs. 

## Obtaining latest version of the software

Visit https://github.com/appl-lab/CuTS for obtaining latest of this software. 

## Package requirements:
* cmake(>=3.10)
* OpenMP
* CUDA(>=10.0)
* MPI (openmpi/3.0.1)

## Reproduce the results in table 3
    
    mkdir build
    cd build
    cmake ..
    make
    cd ..
Now an executable file 'cuts' is generated. 

Run the following commands to reproduce the results in table 3

./build/cuts ./data/email-Enron.txt ./pattern/1.g
./build/cuts ./data/email-Enron.txt ./pattern/2.g
./build/cuts ./data/email-Enron.txt ./pattern/3.g
./build/cuts ./data/email-Enron.txt ./pattern/4.g
./build/cuts ./data/email-Enron.txt ./pattern/5.g
./build/cuts ./data/email-Enron.txt ./pattern/6.g
./build/cuts ./data/email-Enron.txt ./pattern/7.g
./build/cuts ./data/email-Enron.txt ./pattern/8.g

To reproduce the data sets mico, youtube, change the data set path in above commands.