## Overview
The project is modified based on STMatch.

## Build

### Data Preparation

Go to graph_converter directory and compile graph converter
```Shell
cd graph_converter/
make
```

In graph_converter directory, run prepare_data.sh to download graph data and transfer the data format to our system's format. 
The shell script prepare_data.sh will also labelize the downloaded graphs
In prepare_data.sh, we use email-Enron data set as an example.
```Shell
#You can uncomment the link of other graphs in prepare_data.sh \
#if you want to test them and then re-run prepare_data.sh
bash prepare_data.sh  
```

Now you can see some directories in ~/project/data/bin_graph/
```Shell
#it's like "email-Enron"
```

### Compile
Go to home directory and compile the project
```Shell
make clean
#Be careful, don't use make's multi-threads execution like make -j 16. 
make
```

## Test 

### Reproducing the results of Table 3
Take matching q2 in Enron as an example, run it in command line
```Shell
./bin/reint.exe data/bin_graph/email-Enron/snap.txt pattern/2.g
```
The output is ${query graph} ${matching time} ${reuse rate}

### Reproducing the results of Figure 8
Generate synthetic data sets using generate_graph.py
Set the num_nodes and the avraverage_degree. (in our test, num_nodes = 100000 and avraverage_degree = 40, 50, 60) and run it.
```Shell
./generate_graph.py
```
Then comment the downloads link in prepare_data.sh and use the prepare_data.sh to preprocess the synthetic data sets. 
Finally, run the matching command like Table 3.

### Reproducing the results of Figure 9
To get the matching time of only using "Reuse":

```Shell
sed -i 's/inline constexpr bool KEY_NODE=true;/inline constexpr bool KEY_NODE=false;/' src/config.h
make clean
make
./bin/reint.exe data/bin_graph/email-Enron/snap.txt pattern/2.g
```

To get the matching time of using "Reuse + Key nodes":
```Shell
sed -i 's/inline constexpr bool KEY_NODE=false;/inline constexpr bool KEY_NODE=true;/' src/config.h
make clean
make
./bin/reint.exe data/bin_graph/email-Enron/snap.txt pattern/2.g
```
### Reproducing the results of Figure 10
Change the value of MAX_SN to 1 in src/config.h and make again. Run it.
```Shell
sed -i 's/inline constexpr int MAX_SN=8;/inline constexpr int MAX_SN=1;/' src/config.h
make clean
make
./bin/reint.exe data/bin_graph/com-orkut.ungraph/snap.txt pattern/2.g
```
Then change the value of MAX_SN to 2, 4, 8 and repeat.

## Test STMatch, cuTS and EGSM

```Shell
    #Please follow the instructions in this directory to test STMatch, cuTS and EGSM
    
```
