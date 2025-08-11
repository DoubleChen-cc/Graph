#pragma once

#include "graph.h"
#include "pattern.h"
#include "callstack.h"
#include "job_queue.h"

namespace STMatch {
  __global__ void _parallel_match(Graph* dev_graph, Pattern* dev_pattern, 
                            CallStack* dev_callstack, JobQueue* job_queue,  size_t* res,Bucket* hash_table,size_t* gpu_intsec_count, Global_CallStack* global_callstack);
  
  

}
