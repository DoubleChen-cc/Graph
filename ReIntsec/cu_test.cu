#include <string>
#include <iostream>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

int main(int argc, char* argv[]) {

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 1);
  //int clockRate = prop.clockRate;
  STMatch::GraphPreprocessor g(argv[1]);
  STMatch::PatternPreprocessor p(argv[2]);
  // std::vector<int> degree_list;
  // long degree_sum=0;
  // int nodes=g.g.nnodes;
  // for(int i=0;i<nodes;i++){
  //   degree_sum+=(g.g.rowptr[i+1]-g.g.rowptr[i]);
  //   degree_list.push_back(g.g.rowptr[i+1]-g.g.rowptr[i]);
  // }
  // sort(degree_list.begin(),degree_list.end());
  // int mid=g.g.nnodes/2;
  // int avrd=degree_sum/g.g.nnodes;
  // printf("nodes:%d,edges:%llu,degree_sum:%llu,avrd:%d,mid_d:%d,max_d:%d\n",g.g.nnodes,g.g.nedges,degree_sum,avrd,degree_list[mid],degree_list[g.g.nnodes-1]);

  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = JobQueuePreprocessor(g.g, p).to_gpu();
  CallStack* gpu_callstack;

  // allocate the callstack for all warps in global memory
  graph_node_t* slot_storage;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  cout << "global memory usage: " << sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE / 1024.0 / 1024 / 1024 << " GB" << endl;

  std::vector<CallStack> stk(NWARPS_TOTAL);
  std::vector<Global_CallStack> global_stk(NWARPS_TOTAL);

  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    auto& gs = global_stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    memset(gs.intsec_vertices,0,sizeof(gs.intsec_vertices));
    memset(gs.rwlock,0,sizeof(RWLock));
    s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  CHECK(cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack)));
  CHECK(cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice));
  Global_CallStack* gpu_global_callstack; 
  CHECK(cudaMalloc(&gpu_global_callstack, NWARPS_TOTAL * sizeof(Global_CallStack)));
  CHECK(cudaMemcpy(gpu_global_callstack, global_stk.data(), sizeof(Global_CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice));

  size_t* gpu_res;
  CHECK(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL));
  CHECK(cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL));
  size_t* res = new size_t[NWARPS_TOTAL];

  // int* idle_warps;
  // CHECK(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM));
  // CHECK(cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM));

  // int* idle_warps_count;
  // CHECK(cudaMalloc(&idle_warps_count, sizeof(int)));
  // CHECK(cudaMemset(idle_warps_count, 0, sizeof(int)));

  // int* global_mutex;
  // CHECK(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM));
  // CHECK(cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM));

  // bool* stk_valid;
  // CHECK(cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM));
  // CHECK(cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM));
  // size_t free_bytes, total_bytes;
  //   cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
  //   if (result == cudaSuccess) {
  //       std::cout << "Total GPU memory: " << total_bytes/1024/1024/1024 << " GB" << std::endl;
  //       std::cout << "Free GPU memory: " << free_bytes/1024/1024/1024 << " GB" << std::endl;
  //   } else {
  //       std::cerr << "Failed to get memory info: " << cudaGetErrorString(result) << std::endl;
  //   }
    
  Bucket* hash_table;
  CHECK(cudaMalloc(&hash_table,sizeof(Bucket)*HASH_TABLE_SIZE));
  CHECK(cudaMemset(hash_table,0,sizeof(Bucket)*HASH_TABLE_SIZE));
  cout << "hash_table memory usage: " << sizeof(Bucket) * HASH_TABLE_SIZE / 1024.0 / 1024 / 1024 << " GB" << endl;

  size_t* gpu_intsec_count;
  CHECK(cudaMalloc(&gpu_intsec_count, sizeof(size_t) * NWARPS_TOTAL*INTSEC_SIZE));
  CHECK(cudaMemset(gpu_intsec_count, 0, sizeof(size_t) * NWARPS_TOTAL*INTSEC_SIZE));
  size_t* intsec_count = new size_t[NWARPS_TOTAL*INTSEC_SIZE];

  // size_t* gpu_clock_count;
  // cudaMalloc(&gpu_clock_count, sizeof(size_t) * NWARPS_TOTAL*3);
  // cudaMemset(gpu_clock_count, 0, sizeof(size_t) * NWARPS_TOTAL*3);
  // size_t* clock_count = new size_t[NWARPS_TOTAL*3];

  // int* gpu_compu_set_count;
  // cudaMalloc(&gpu_compu_set_count, sizeof(int) * NWARPS_TOTAL*3);
  // cudaMemset(gpu_compu_set_count, 0, sizeof(int) * NWARPS_TOTAL*3);
  // size_t* compu_set_count = new size_t[NWARPS_TOTAL*3];


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //cout << "shared memory usage: " << sizeof(Graph) << " " << sizeof(Pattern) << " " << sizeof(JobQueue) << " " << sizeof(CallStack) * NWARPS_PER_BLOCK << " " << NWARPS_PER_BLOCK * 33 * sizeof(int) << " Bytes" << endl;

  // _parallel_match << <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, idle_warps_count, global_mutex, hash_table,gpu_intsec_count,gpu_clock_count,gpu_compu_set_count);
  // 启动内核

  // void* args[] = {gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, hash_table,gpu_intsec_count,gpu_global_callstack};
  //   cudaError_t err = cudaLaunchKernel((const void*)&_parallel_match,
  //                                      (dim3)(GRID_DIM,1,1),(dim3)(BLOCK_DIM,1,1),
  //                                      args);

  //   if (err != cudaSuccess) {
  //       std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  //   }

    _parallel_match << <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res,hash_table,gpu_intsec_count,gpu_global_callstack);
    
  cudaDeviceSynchronize();


  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("matching time: %f ms\n", milliseconds);

  cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);
  cudaMemcpy(intsec_count, gpu_intsec_count, sizeof(size_t) * NWARPS_TOTAL*INTSEC_SIZE, cudaMemcpyDeviceToHost);
  // cudaMemcpy(clock_count, gpu_clock_count, sizeof(size_t) * NWARPS_TOTAL*3, cudaMemcpyDeviceToHost);
  // cudaMemcpy(compu_set_count, gpu_compu_set_count, sizeof(int) * NWARPS_TOTAL*3, cudaMemcpyDeviceToHost);

  unsigned long long tot_count = 0;
  unsigned long long tot_intsec_count[INTSEC_SIZE] = {0,0};
  // float avr_clock_count[3] = {0.0,0.0,0.0};
  // int avr_compu_set_count[3]={0,0,0};
  for (int i=0; i<NWARPS_TOTAL; i++) {
    tot_count += res[i];
    for(int j=0;j<INTSEC_SIZE;j++){
       tot_intsec_count[j] += intsec_count[i*INTSEC_SIZE+j];
     }
    // for(int j=0;j<3;j++){
    //    float ever=avr_clock_count[j];
    //    if(clock_count[i*3+j]>ever) avr_clock_count[j] =ever + (float)(clock_count[i*3+j]-ever)/(i+1);
    //    else avr_clock_count[j] =ever - (float)(ever-clock_count[i*3+j])/(i+1);
    //    avr_compu_set_count[j]+=compu_set_count[i*3+j];
    //  }
  }
  // float avr_t[3]={0.0,0.0,0.0};
  // for(int i=0;i<3;i++){
  //   size_t tmp=avr_clock_count[i];
  //   // avr_t[i]=(float)tmp/(float)clockRate/1000*1000.0;
  // }

  if(!LABELED) tot_count = tot_count * p.PatternMultiplicity;
  float reuse_rate;
 // 检查除数是否为零
  if (tot_intsec_count[0] == 0) {
      // 处理除零情况，例如设置为0或打印错误信息
      reuse_rate = 0.0f;
      // 或者：fprintf(stderr, "Error: division by zero\n");
  } else {
      reuse_rate = static_cast<float>(tot_intsec_count[1]) / tot_intsec_count[0];
  }

  // 确保格式符与变量类型匹配
  printf("%s\t%f\t%f\n", argv[2], milliseconds, reuse_rate);
  //cout << "count: " << tot_count << endl;
  // printf("%f,%f,%f\n",avr_t[0],avr_t[1],avr_t[2]);
  // printf("%f,%f,%f\n",(float)avr_compu_set_count[0]/NWARPS_TOTAL,(float)avr_compu_set_count[1]/NWARPS_TOTAL,(float)avr_compu_set_count[2]/NWARPS_TOTAL);


  //string fstring="result.txt";
  //std::ofstream outputFile(fstring,std::ios::app);
  //outputFile<<argv[1]<<' '<<argv[2]<<' '<<milliseconds<<' '<<tot_count<<' '<<tot_intsec_count[0]<<' '<<tot_intsec_count[1]<<endl;
  //outputFile.close();
  return 0;

}

