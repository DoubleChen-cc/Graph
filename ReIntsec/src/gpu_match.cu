
#include "gpu_match.cuh"
#include <cuda.h>

#define UNROLL_SIZE(l) (l > 0 ? UNROLL: 1) 

namespace STMatch {
  struct StealingArgs {
    int* idle_warps;
    int* idle_warps_count;
    int* global_mutex;
    int* local_mutex;
    CallStack* global_callstack;
  };

      //读锁获取，如果正在被写返回-1，否则返回1
  __device__ int acquireReadLock(RWLock* lock) {
    int lockStatus=0;
    if(threadIdx.x%WARP_SIZE==0){
      lockStatus=atomicCAS(&lock->writer, 0, 0);
    }
    lockStatus = __shfl_sync(0xFFFFFFFF, lockStatus, 0);
    if(lockStatus!=0)return -1;
    else{
      if(threadIdx.x%WARP_SIZE==0){
        atomicAdd(&lock->readers, 1);
      }
      __syncwarp();
      return 1;
    }
  }

  // 读锁释放
  __device__ void releaseReadLock(RWLock* lock) {
    if(threadIdx.x%WARP_SIZE==0){
      atomicAdd(&lock->readers, -1);
    }
    __syncwarp();
  }

  //如果正在被读返回-1，如果正在被写返回1，如果没读没写返回0；
  __device__ int acquireWriteLock(RWLock* lock){
    int writeStatus=0;
    if(threadIdx.x%WARP_SIZE==0){
      writeStatus=atomicCAS(&lock->writer, 0, 1);
    }
    writeStatus = __shfl_sync(0xFFFFFFFF, writeStatus, 0);
    if(writeStatus==0){
      int readStatus=0;
      if(threadIdx.x%WARP_SIZE==0){
        readStatus=atomicCAS(&lock->readers, 0, 0);
      }
      readStatus = __shfl_sync(0xFFFFFFFF, readStatus, 0);
      if(readStatus!=0){
        if(threadIdx.x%WARP_SIZE==0){
          atomicExch(&lock->writer, 0);
        }
        __syncwarp();
        return -1;
      }
      else return 0;
    }else return 1;
  }

  __device__ void releaseWriteLock(RWLock* lock) {
    if(threadIdx.x%WARP_SIZE==0){
      atomicExch(&lock->writer, 0);
    }
    __syncwarp();
  }

  // __device__ int acquireReadLock1(RWLock* lock) {
  //   if(threadIdx.x%WARP_SIZE==0){
  //     while(atomicCAS(&lock->writer, 0, 0)!=0){
  //       ;
  //     }
  //     atomicAdd(&lock->readers, 1);
  //   }
  //   __syncwarp();
  //   return 1;
  // }

  // // 读锁释放
  // __device__ void releaseReadLock1(RWLock* lock) {
  //   if(threadIdx.x%WARP_SIZE==0){
  //     atomicAdd(&lock->readers, -1);
  //   }
  //   __syncwarp();
  // }

  // __device__ int acquireWriteLock1(RWLock* lock){
  //   if(threadIdx.x%WARP_SIZE==0){
  //     while(atomicCAS(&lock->writer, 0, 1)!=0){
  //       ;
  //     }
  //     while(atomicCAS(&lock->readers, 0, 0)!=0){
  //       ;
  //     } 
  //   }
  //   __syncwarp();
  //   return 0;
  // }

  // __device__ void releaseWriteLock1(RWLock* lock) {
  //   if(threadIdx.x%WARP_SIZE==0){
  //     atomicExch(&lock->writer, 0);
  //   }
  //   __syncwarp();
  // }
  __forceinline__ __device__ void lock(int* mutex) {
    while (atomicCAS((int*)mutex, 0, 1) != 0) {
    }
  }
  __forceinline__ __device__ void unlock(int* mutex) {
    atomicExch((int*)mutex, 0);
  }
 __device__ void CpyMem(graph_node_t* dst, graph_node_t* src, unsigned size){
  int tid =  threadIdx.x%WARP_SIZE;
  for(int i=tid;i<size;i+=WARP_SIZE){
    dst[i]=src[i];
  }
  __syncwarp();
}


__device__ graph_node_t CpyRM(graph_node_t* dst, graph_node_t* src, unsigned size,CallStack* stk, int slot_idx,int uiter){
    int end=0;
    int tid=threadIdx.x%WARP_SIZE;
    for(int i=tid;i<((size+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE;i+=WARP_SIZE){
      int flag=1;
      if(i<size){
        for(int j=1;j<=stk->slot_size[slot_idx][uiter];j++){
          if(src[i]==stk->slot_storage[slot_idx][uiter][j]){
            flag=0;
            break;
          }
        }
      }else{
        flag=0;
      }
      __syncwarp();
      int predicate = __ballot_sync(0xFFFFFFFF, flag);
      if(flag==1)dst[end+__popc(predicate&((1<<tid)-1))]=src[i];
      __syncwarp();
      end+=__popc(predicate);
      __syncwarp();
    }
    return end;       
}
  __device__ uint32_t MurmurHash3(const void *key, size_t length, uint32_t seed){
    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    uint32_t h = seed;
    const uint8_t *data = static_cast<const uint8_t*>(key);

    while (length > 0) {
        uint32_t k;

        // Unpack four bytes from key into a 32-bit number k
        if (length >= 4) {
            k = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
            length -= 4;
        } else {
            k = 0;
            for (size_t i = 0; i < length; i++) {
                k |= (data[i] << (8 * (3 - i)));
            }
            length = 0;
        }

        // Mix 4 bytes into the hash
        k *= m;
        k ^= k >> r;
        k *= m;

        // Accumulate the hash value
        h *= m;
        h ^= k;
    }

    // Finalization
    h ^= h >> 13;
    h *= 5;
    h ^= h >> 3;
    h *= 5;
    h ^= h >> 16;

    return h;
  }
  __device__ unsigned MyHash(graph_node_t t0, graph_node_t t1){
      uint32_t seed=0x9e3779b9;
      uint32_t p=402653171;
      size_t key;
      if(t0<t1){
          key=t0;
          key=key*p+t1;
      }else{
          key=t1;
          key=key*p+t0;
      }
      size_t length=8;
      unsigned value=MurmurHash3(&key,length,seed)%HASH_TABLE_SIZE;
      return value;
  }
  __device__ int CheckRepeated(graph_node_t t0, graph_node_t t1,unsigned index, Global_CallStack* global_stk, Bucket* hash_table){
    int tid=threadIdx.x%WARP_SIZE;
    graph_node_t temp_t;
    if(t1<t0){
      temp_t=t0;
      t0=t1;
      t1=temp_t;
    }
    for(int i=0;i<CHECK_BUCKET_NUM;i++){
      int flag=0;
      if(tid<BUCKET_SIZE){
        hash_t value=hash_table[index+i].value[tid];
        // unsigned v0_xor_v1=value>>(HASH_SL_BIT+HASH_K_BIT+HASH_GL_BIT);
        // if(((t0^t1)%0xffff)==v0_xor_v1){
        //   flag=1;
        // }
        graph_node_t v0=value>>(HASH_V_BIT+HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT);
        graph_node_t v1=(value&0x000003fffff00000)>>(HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT);
        if(v0==t0 && v1==t1){
          flag=1;
        }
      }
      __syncwarp();
      int predicate = __ballot_sync(0xFFFFFFFF, flag);
      if(predicate){
        for(int j=0;j<BUCKET_SIZE;j++){
          if((predicate&(1<<j))){
            hash_t value=hash_table[index+i].value[j];
            int global_stk_idx=(value&((1<<(HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT))-1))>>(HASH_SL_BIT+HASH_K_BIT);
            int slot_idx=(value&((1<<(HASH_SL_BIT+HASH_K_BIT))-1))>>HASH_K_BIT;
            int k=value&((1<<HASH_K_BIT)-1);
            graph_node_t vv0=global_stk[global_stk_idx].intsec_vertices[slot_idx][k][0];
            graph_node_t vv1=global_stk[global_stk_idx].intsec_vertices[slot_idx][k][1];
            if(t0==vv0 && t1==vv1)return ((i<<3)|j);
          }
        }
      }
    }
    return -1;
  }

  __device__ void InsertHashTable(graph_node_t t0, graph_node_t t1, unsigned stkidx,unsigned slot_idx, unsigned k, Bucket* hash_table){
    unsigned t_index=MyHash(t0,t1);
    graph_node_t temp_t;
    if(t1<t0){
      temp_t=t0;
      t0=t1;
      t1=temp_t;
    }
    unsigned index=t_index/BUCKET_SIZE;
    // unsigned t0_xor_t1=(t0^t1)%0xffff;
    hash_t idx=stkidx, sl=slot_idx, kk=k,v0=t0,v1=t1;
    // hash_t value=(t0_xor_t1<<(HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT))|(idx<<(HASH_SL_BIT+HASH_K_BIT))|(sl<<HASH_K_BIT)|kk;
    hash_t value=(v0<<(HASH_V_BIT+HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT))|(v1<<(HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT))|(idx<<(HASH_SL_BIT+HASH_K_BIT))|(sl<<HASH_K_BIT)|kk;
    int tid=threadIdx.x % WARP_SIZE;

    for(int i=0;i<CHECK_BUCKET_NUM;i++){
      int flag=0;
      if(tid<BUCKET_SIZE){
        if(hash_table[index+i].value[tid]==0){
          flag=1;
        }
      }
      __syncwarp();
      int predicate = __ballot_sync(0xFFFFFFFF, flag);
      if(predicate){
        if(tid==0){
          for(int j=0;j<BUCKET_SIZE;j++){
            if((predicate&(1<<j))){
              hash_table[index+i].value[j]=value;
              break;
            }
          }
        }
        __syncwarp();
        break;
      }
      else{
        if(i==CHECK_BUCKET_NUM-1)
          hash_table[index].value[t_index%BUCKET_SIZE]=value;
      }
    }
  }

  // __device__ bool trans_layer(CallStack& _target_stk, CallStack& _cur_stk, Pattern* _pat, int _k, int ratio = 2) {
  //   if (_target_stk.level <= _k)
  //     return false;

  //   int num_left_task = _target_stk.slot_size[_pat->rowptr[_k]][_target_stk.uiter[_k]] -
  //     (_target_stk.iter[_k] + _target_stk.uiter[_k + 1] + 1);
  //   if (num_left_task <= 0)
  //     return false;

  //   int stealed_start_idx_in_target = _target_stk.iter[_k] + _target_stk.uiter[_k + 1] + 1 + num_left_task / ratio;

  //   _cur_stk.slot_storage[_pat->rowptr[0]][_target_stk.uiter[0]][_target_stk.iter[0] + _target_stk.uiter[1]] = _target_stk.slot_storage[_pat->rowptr[0]][_target_stk.uiter[0]][_target_stk.iter[0] + _target_stk.uiter[1]];
  //   _cur_stk.slot_storage[_pat->rowptr[0]][_target_stk.uiter[0]][_target_stk.iter[0] + _target_stk.uiter[1] + JOB_CHUNK_SIZE] = _target_stk.slot_storage[_pat->rowptr[0]][_target_stk.uiter[0]][_target_stk.iter[0] + _target_stk.uiter[1] + JOB_CHUNK_SIZE];

  //   for (int i = 1; i < _k; i++) {
  //     _cur_stk.slot_storage[_pat->rowptr[i]][_target_stk.uiter[i]][_target_stk.iter[i] + _target_stk.uiter[i + 1]] = _target_stk.slot_storage[_pat->rowptr[i]][_target_stk.uiter[i]][_target_stk.iter[i] + _target_stk.uiter[i + 1]];
  //   }

  //   for (int r = _pat->rowptr[_k]; r < _pat->rowptr[_k + 1]; r++) {
  //     for (int u = 0; u < UNROLL_SIZE(_k); u++) {
  //       int loop_end = _k == 0 ? JOB_CHUNK_SIZE * 2 : _target_stk.slot_size[r][u];
  //       for (int t = 0; t < loop_end; t++) {
  //         _cur_stk.slot_storage[r][u][t] = _target_stk.slot_storage[r][u][t];
  //       }
  //     }
  //   }

  //   for (int l = 0; l < _k; l++) {
  //     _cur_stk.iter[l] = _target_stk.iter[l];
  //     _cur_stk.uiter[l] = _target_stk.uiter[l];
  //     for (int s = _pat->rowptr[l]; s < _pat->rowptr[l + 1]; s++) {
  //       if (s > _pat->rowptr[l]) {
  //         for (int u = 0; u < UNROLL; u++) {
  //           _cur_stk.slot_size[s][u] = _target_stk.slot_size[s][u];
  //         }
  //       }
  //       else {
  //         for (int u = 0; u < UNROLL_SIZE(l); u++) {
  //           if (u == _cur_stk.uiter[l])
  //             _cur_stk.slot_size[_pat->rowptr[l]][u] = _target_stk.iter[l] + 1;
  //           else
  //             _cur_stk.slot_size[_pat->rowptr[l]][u] = 0;
  //         }
  //       }
  //     }
  //   }

  //   // copy
  //   for (int i = stealed_start_idx_in_target - _target_stk.iter[_k]; i < UNROLL_SIZE(_k + 1); i++) {
  //     _target_stk.slot_size[_pat->rowptr[_k + 1]][i] = 0;
  //   }

  //   for (int s = _pat->rowptr[_k]; s < _pat->rowptr[_k + 1]; s++) {
  //     if (s == _pat->rowptr[_k]) {
  //       for (int u = 0; u < UNROLL_SIZE(_k); u++) {
  //         if (u == _target_stk.uiter[_k])
  //           _cur_stk.slot_size[s][u] = _target_stk.slot_size[s][u];
  //         else
  //           _cur_stk.slot_size[s][u] = 0;
  //       }
  //     }
  //     else {
  //       for (int u = 0; u < UNROLL_SIZE(_k); u++) {
  //         _cur_stk.slot_size[s][u] = _target_stk.slot_size[s][u];
  //       }
  //     }
  //   }

  //   _cur_stk.uiter[_k] = _target_stk.uiter[_k];
  //   _cur_stk.iter[_k] = stealed_start_idx_in_target;
  //   _target_stk.slot_size[_pat->rowptr[_k]][_target_stk.uiter[_k]] = stealed_start_idx_in_target;
  //   // copy
  //   for (int l = _k + 1; l < _pat->nnodes - 1; l++) {
  //     _cur_stk.iter[l] = 0;
  //     _cur_stk.uiter[l] = 0;
  //     for (int s = _pat->rowptr[l]; s < _pat->rowptr[l + 1]; s++) {
  //       for (int u = 0; u < UNROLL_SIZE(l); u++) {
  //         _cur_stk.slot_size[s][u] = 0;
  //       }
  //     }
  //   }
  //   _cur_stk.iter[_pat->nnodes - 1] = 0;
  //   _cur_stk.uiter[_pat->nnodes - 1] = 0;
  //   for (int u = 0; u < UNROLL_SIZE(_pat->nnodes - 1); u++) {
  //     _cur_stk.slot_size[_pat->rowptr[_pat->nnodes - 1]][u] = 0;
  //   }
  //   _cur_stk.level = _k + 1;
  //   return true;
  // }

  // __device__ bool trans_skt(CallStack* _all_stk, CallStack* _cur_stk, Pattern* pat, StealingArgs* _stealing_args) {

  //   int max_left_task = 0;
  //   int stk_idx = -1;
  //   int at_level = -1;

  //   for (int level = 0; level < STOP_LEVEL; level++) {
  //     for (int i = 0; i < NWARPS_PER_BLOCK; i++) {

  //       if (i == threadIdx.x / WARP_SIZE)
  //         continue;
  //       lock(&(_stealing_args->local_mutex[i]));

  //       int left_task = _all_stk[i].slot_size[pat->rowptr[level]][_all_stk[i].uiter[level]] -
  //         (_all_stk[i].iter[level] + _all_stk[i].uiter[level + 1] + 1);
  //       if (left_task > max_left_task) {
  //         max_left_task = left_task;
  //         stk_idx = i;
  //         at_level = level;
  //       }
  //       unlock(&(_stealing_args->local_mutex[i]));
  //     }
  //     if (stk_idx != -1)
  //       break;
  //   }

  //   if (stk_idx != -1) {
  //     bool res;
  //     lock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
  //     lock(&(_stealing_args->local_mutex[stk_idx]));
  //     res = trans_layer(_all_stk[stk_idx], *_cur_stk, pat, at_level);

  //     unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
  //     unlock(&(_stealing_args->local_mutex[stk_idx]));
  //     return res;
  //   }
  //   return false;
  // }


  __forceinline__ __device__ graph_node_t path(CallStack* stk, Pattern* pat, int level, int k) {
    if (level > 0){
      int stack_num=(pat->new_set_ops[pat->rowptr[level]]&0xf00)>>8;
      int slot_idx;
      if((pat->new_set_ops[pat->rowptr[level]]&0x1000)&&level>=1)
        slot_idx=pat->slotptr[pat->rowptr[level]]+stack_num;
      else slot_idx=pat->slotptr[pat->rowptr[level]]+stk->giter[level]%stack_num;
      return stk->slot_storage[slot_idx][stk->uiter[level]][stk->iter[level] + k];
    }
    else {
      return stk->slot_storage[0][stk->uiter[0]][stk->iter[0] + k + (level + 1) * JOB_CHUNK_SIZE];
    }
  }

  __forceinline__ __device__ graph_node_t* path_address(CallStack* stk, Pattern* pat, int level, int k) {
    if (level > 0){
      int stack_num=(pat->new_set_ops[pat->rowptr[level]]&0xf00)>>8;
      int slot_idx;
      if((pat->new_set_ops[pat->rowptr[level]]&0x1000)&&level>=1)
        slot_idx=pat->slotptr[pat->rowptr[level]]+stack_num;
      else slot_idx=pat->slotptr[pat->rowptr[level]]+stk->giter[level]%stack_num;
      return &(stk->slot_storage[slot_idx][stk->uiter[level]][stk->iter[level] + k]);
    }
      
    else {
      return &(stk->slot_storage[0][stk->uiter[0]][stk->iter[0] + k + (level + 1) * JOB_CHUNK_SIZE]);
    }
  }

  typedef struct {
    graph_node_t* set1[UNROLL], * set2[UNROLL], * res[UNROLL];
    set_size_t set1_size[UNROLL], set2_size[UNROLL], * res_size[UNROLL];
    graph_node_t ub[UNROLL];
    graph_node_t key_nodes[UNROLL][WARP_SIZE];
    //bitarray32 label;
    Graph* g;
    int num_sets;
    bool un_intsec=false;
    int level;
    Pattern* pat;
    bool repeated[UNROLL];
    bool origin[UNROLL];

  } Arg_t;

  template<typename DATA_T, typename SIZE_T>
  __forceinline__ __device__
    bool bsearch_exist(DATA_T* set2, SIZE_T set2_size, DATA_T target) {
    if (set2_size <= 0) return false;
    int mid;
    int low = 0;
    int high = set2_size - 1;
    while (low <= high) {
      mid = (low + high) / 2;
      if (target == set2[mid]) {
        return true;
      }
      else if (target > set2[mid]) {
        low = mid + 1;
      }
      else {
        high = mid - 1;
      }
    }
    return false;
  }

  template<typename DATA_T, typename SIZE_T>
  __forceinline__ __device__
    SIZE_T upper_bound(DATA_T* set2, SIZE_T set2_size, DATA_T target) {
    int i, step;
    int low = 0;
    while (set2_size > 0) {
      i = low;
      step = set2_size / 2;
      i += step;
      if (target > set2[i]) {
        low = ++i; set2_size -= step + 1;
      }
      else {
        set2_size = step;
      }
    }
    return low;
  }

  template<typename DATA_T, typename SIZE_T>
__forceinline__ __device__
SIZE_T lower_bound(DATA_T* set2, SIZE_T set2_size, DATA_T target) {
    int i, step;
    int low = 0;
    while (set2_size > 0) {
        i = low;
        step = set2_size / 2;
        i += step;
        if (target >= set2[i]) {
            low = ++i; set2_size -= step + 1;
        }
        else {
            set2_size = step;
        }
    }
    return low-1;
}

  __forceinline__ __device__
    void prefix_sum(int* _input, int input_size) {

    int thid = threadIdx.x % WARP_SIZE;
    int offset = 1;
    int last_element = _input[input_size - 1];
    // build sum in place up the tree
    for (int d = (WARP_SIZE >> 1); d > 0; d >>= 1) {
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        _input[bi] += _input[ai];
      }
      offset <<= 1;
    }
    if (thid == 0) { _input[WARP_SIZE - 1] = 0; } // clear the last element
     // traverse down tree & build scan
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
      offset >>= 1;
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        int t = _input[ai];
        _input[ai] = _input[bi];
        _input[bi] += t;
      }
    }
    __syncwarp();

    if (thid >= input_size - 1)
      _input[thid + 1] = _input[input_size - 1] + last_element;
  }


  template<bool DIFF>
  __device__ void compute_set(Arg_t* arg) {
    __shared__ graph_node_t size_psum[NWARPS_PER_BLOCK][WARP_SIZE + 1];
    __shared__ int end_pos[NWARPS_PER_BLOCK][UNROLL];

    int wid = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x % WARP_SIZE;
    
    if (tid < arg->num_sets) {
      if(DIFF||(!DIFF && arg->un_intsec) ||(!DIFF && (arg->origin[tid]==true)))arg->set1_size[tid] = upper_bound(arg->set1[tid], arg->set1_size[tid], arg->ub[tid]);
      size_psum[wid][tid] = arg->set1_size[tid];
      // if((blockIdx.x * blockDim.x + threadIdx.x)<arg->num_sets && arg->level==1 && DIFF==false && (arg->pat->set_ops[1] & 0x40)) printf("wid:%d,tid:%d,set1_size:%d\n",wid,tid,arg->set1_size[tid]);
      end_pos[wid][tid] = 0;
    }
    else {
      size_psum[wid][tid] = 0;
    }
    __syncwarp();

    prefix_sum(&size_psum[wid][0], arg->num_sets);
    // if((blockIdx.x * blockDim.x + threadIdx.x)==0 && arg->level==1 && DIFF==false && (arg->pat->set_ops[1] & 0x40)){
    //   for(int i=0;i<arg->num_sets+1;i++){
    //     printf("size_psum[0][%d]:%d\n",i,size_psum[0][i]);
    //   }
    // }
    __syncwarp();

    bool still_loop = true;
    int slot_idx = 0;
    int offset = 0;
    int predicate;

    for (int idx = tid; (idx < ((size_psum[wid][WARP_SIZE] > 0) ? (((size_psum[wid][WARP_SIZE] - 1) / WARP_SIZE + 1) * WARP_SIZE) : 0) && still_loop); idx += WARP_SIZE) {
      predicate = 0;

      if (idx < size_psum[wid][WARP_SIZE]) {

        while (idx >= size_psum[wid][slot_idx + 1]) {
          slot_idx++;
        }
        offset = idx - size_psum[wid][slot_idx];

        //bitarray32 lb = arg->g->vertex_label[arg->set1[slot_idx][offset]];
//DIFF=true时检查前offset的v1是不是“对应”v0的邻居，不是的话加入slot_storage[i][slot_idx][end_pos]中？？？（不懂）可能与set_ops的值有关
      if(KEY_NODE && !DIFF && !arg->un_intsec){
        if(arg->set2_size[slot_idx]<=WARP_SIZE*16)
          predicate = (DIFF ^ bsearch_exist(arg->set2[slot_idx], arg->set2_size[slot_idx], arg->set1[slot_idx][offset]));
        else{
            graph_node_t index=lower_bound(&arg->key_nodes[slot_idx][0],WARP_SIZE,arg->set1[slot_idx][offset]);
            int step=arg->set2_size[slot_idx]/WARP_SIZE;
            if(index==-1)predicate=0;
            else if(index<31){
              predicate =  bsearch_exist(&arg->set2[slot_idx][index*step],step, arg->set1[slot_idx][offset]);
            }else{
              predicate =  bsearch_exist(&arg->set2[slot_idx][index*step],arg->set2_size[slot_idx]-index*step, arg->set1[slot_idx][offset]);
            }
        }
      }else{
        predicate = (DIFF ^ bsearch_exist(arg->set2[slot_idx], arg->set2_size[slot_idx], arg->set1[slot_idx][offset]));
      }

      }
      else {
        slot_idx = arg->num_sets;
        still_loop = false;
      }

      still_loop = __shfl_sync(0xFFFFFFFF, still_loop, 31);
      predicate = __ballot_sync(0xFFFFFFFF, predicate);

      bool cond = ((arg->level<arg->pat->nnodes-2)||((arg->level==arg->pat->nnodes-2)&&!arg->un_intsec && !DIFF))&&(predicate & (1 << tid));
      // bool cond = predicate & (1 << tid);
      graph_node_t res_tmp;
      if (cond ) {
        res_tmp = arg->set1[slot_idx][offset];
      }

      int prev_idx = ((idx / WARP_SIZE == size_psum[wid][slot_idx] / WARP_SIZE) ? size_psum[wid][slot_idx] % WARP_SIZE : 0);
      //prev_idx用于查看当前线程前面属于上一个slot_idx的线程
      //__popc(predicate & ((1 << tid) - (1 << prev_idx)))获得从prev_idx到当前线程中1的个数
      if (cond&& (!arg->repeated[slot_idx])) {
        arg->res[slot_idx][end_pos[wid][slot_idx] + __popc(predicate & ((1 << tid) - (1 << prev_idx)))] = res_tmp;
      }

      if (slot_idx < __shfl_down_sync(0xFFFFFFFF, slot_idx, 1)) {
        end_pos[wid][slot_idx] += __popc(predicate & ((1 << (tid + 1)) - (1 << prev_idx)));
      }
      //__popc(predicate & (0xFFFFFFFF - (1 << prev_idx) + 1))从prev_idx到warp最后一个线程中1的个数
      else if (tid == WARP_SIZE - 1 && slot_idx < arg->num_sets) {
        end_pos[wid][slot_idx] += __popc(predicate & (0xFFFFFFFF - (1 << prev_idx) + 1));
      }
    }
    __syncwarp();
    if (tid < arg->num_sets) {
      if(!arg->repeated[tid]){
        *(arg->res_size[tid]) = end_pos[wid][tid];
      }
    }
    __syncwarp();
  }

  __forceinline__ __device__ void get_job(JobQueue* q, graph_node_t& cur_pos, graph_node_t& njobs) {
    lock(&(q->mutex));
    cur_pos = q->cur;
    q->cur += JOB_CHUNK_SIZE;
    if (q->cur > q->length) q->cur = q->length;
    njobs = q->cur - cur_pos;
    unlock(&(q->mutex));
  }

  __device__ void extend(Graph* g, Pattern* pat, CallStack* stk, JobQueue* q, pattern_node_t level,Bucket* hash_table, CallStack* global_stk,Global_CallStack* global_stk_intsec) {

    __shared__ Arg_t arg[NWARPS_PER_BLOCK];
    __shared__ graph_node_t insert_vertices[NWARPS_PER_BLOCK][UNROLL][2];
    
    int tid = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int global_wid = global_tid / WARP_SIZE;
    memset(&arg[wid].repeated[0],false,sizeof(bool)*UNROLL); 
    memset(&arg[wid].origin[0],false,sizeof(bool)*UNROLL); 



    if (level == 0) {
      graph_node_t cur_job, njobs;

      // TODO: change to warp
      for (int k = 0; k < UNROLL_SIZE(level); k++) {
        if (threadIdx.x % WARP_SIZE == 0) {
          get_job(q, cur_job, njobs); 

          for (size_t i = 0; i < njobs; i++) {
            for (int j = 0; j < 2; j++) {
              stk->slot_storage[0][k][i + JOB_CHUNK_SIZE * j] = (q->q[cur_job + i].nodes)[j];
            }
          }
          stk->slot_size[0][k] = njobs;
          //stk->slot_size[0][k][1] = njobs;
        }
        __syncwarp();
      }
    }
    else {
      
      arg[wid].g = g;
      arg[wid].num_sets = UNROLL_SIZE(level);
      
      
      int stk_num=(pat->new_set_ops[pat->rowptr[level-1]]&0xf00)>>8;
      int rmslot=pat->slotptr[pat->rowptr[level-1]]+(((pat->new_set_ops[pat->rowptr[level-1]]&0x1000)&&level>=2)?stk_num:0);
      //int remaining = stk->slot_size[rmslot][stk->uiter[level - 1]][0] - stk->iter[level - 1];
      int remaining = stk->slot_size[rmslot][stk->uiter[level - 1]] - stk->iter[level - 1];
      if (remaining >= 0 && UNROLL_SIZE(level) > remaining) {
        arg[wid].num_sets = remaining;
      }
      for (int i = pat->rowptr[level]; i < pat->rowptr[level + 1]; i++) {

        // compute ub based on pattern->partial
        if (!LABELED) {
          graph_node_t ub = ((i == pat->rowptr[level]) ? INT_MAX : -1);
          if (pat->partial[i] != 0) {

            // compute ub with nodes after start_level until previous level
            for (pattern_node_t k = 1; k < level - 1; k++) {
              if ((pat->partial[i] & (1 << (k + 1))) && ((i == pat->rowptr[level]) ^ (ub < path(stk, pat, k, stk->uiter[k + 1])))) ub = path(stk, pat, k, stk->uiter[k + 1]);
            }
            // compute ub with nodes in the previous level
            for (pattern_node_t k = 0; k < arg[wid].num_sets; k++) { 
              arg[wid].ub[k] = ub;
              int prev_level = (level > 1 ? 2 : 1);
              int prev_iter = (level > 1 ? stk->uiter[1] : k);
              // compute ub with the first few nodes before start_level
              for (pattern_node_t j = 0; j < prev_level; j++) {
                if ((pat->partial[i] & (1 << j)) && ((i == pat->rowptr[level]) ^ (arg[wid].ub[k] < path(stk, pat, j - 1, prev_iter)))) arg[wid].ub[k] = path(stk, pat, j - 1, prev_iter);
              }

              if ((pat->partial[i] & (1 << level)) && ((i == pat->rowptr[level]) ^ (arg[wid].ub[k] < path(stk, pat, level - 1, k)))) arg[wid].ub[k] = path(stk, pat, level - 1, k);
              if (arg[wid].ub[k] == -1) arg[wid].ub[k] = INT_MAX;
            }
          }
          else {
            for (pattern_node_t k = 0; k < arg[wid].num_sets; k++) {
              arg[wid].ub[k] = INT_MAX;
            }
          }
        } 
        else {
          for (pattern_node_t k = 0; k < arg[wid].num_sets; k++) {
            arg[wid].ub[k] = INT_MAX;
          }
        }

        //arg[wid].label = pat->slot_labels[i]; 

        if (pat->new_set_ops[i] & 0x20) {//N(V(l-1))作为第二个操作数，做差
            int stack_num=(pat->new_set_ops[i]&0xf00)>>8;
            pattern_node_t cur_slot=pat->slotptr[i]+stk->giter[level]%stack_num;
            if(pat->new_set_ops[i]&0x80000){
              int cpy_idx=(pat->new_set_ops[i]&0x7c000)>>14;
              for(int k=0;k<arg[wid].num_sets;k++){
                int size=stk->slot_size[cpy_idx][k];
                CpyMem(&stk->slot_storage[cur_slot][k][0],&stk->slot_storage[cpy_idx][k][0],size);
                stk->slot_size[cur_slot][k]=size;
                //stk->slot_size[cur_slot][k][1]=size;
              }
              for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++){
                stk->slot_size[cur_slot][k] = 0;
                //stk->slot_size[cur_slot][k][1] = 0;
              }    
              break; 
            }
          
          for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {

            arg[wid].set2[k] = NULL;
            arg[wid].set2_size[k] = 0;
            if (!EDGE_INDUCED) {
              graph_node_t t = path(stk, pat, level - 2, ((level > 1) ? stk->uiter[level - 1] : k));
              arg[wid].set2[k] = &g->colidx[g->rowptr[t]];
              arg[wid].set2_size[k] = (set_size_t)(g->rowptr[t + 1] - g->rowptr[t]);
            }
            graph_node_t t = path(stk, pat, level - 1, k); 
            arg[wid].set1[k] = &g->colidx[g->rowptr[t]];
            arg[wid].res[k] = &(stk->slot_storage[cur_slot][k][0]);
            arg[wid].set1_size[k] = (set_size_t)(g->rowptr[t + 1] - g->rowptr[t]);
            arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k]);
            //if(pat->new_set_ops[i]&0x2000)stk->vertices[cur_slot][k]=t;
            if(pat->new_set_ops[i]&0x2000)stk->slot_storage[cur_slot][k][0]=t;
          }
          // arg[wid].cached = (level > 1);
          arg[wid].level = level;
          arg[wid].pat = pat;
          //if(pat->new_set_ops[i]&0x2000)continue;
          //0x2000的栈里[0]储存做交集的t，[1:]储存CpyRM时需要RM的顶点。
          if(pat->new_set_ops[i]&0x2000){
            int unrollIdx = threadIdx.x % WARP_SIZE;
            if(unrollIdx < arg[wid].num_sets)
              stk->slot_size[cur_slot][unrollIdx]=0;
            for (int j = level-1, s=1; j >= -1; j--,s++)
            {  
              if(unrollIdx < arg[wid].num_sets)
              {
                if(level==1){
                  stk->slot_storage[cur_slot][unrollIdx][s] = path(stk, pat, j, unrollIdx);
                  stk->slot_size[cur_slot][unrollIdx]++;
                }
                else{
                  if(j==level-1){
                    stk->slot_storage[cur_slot][unrollIdx][s] = path(stk, pat, j, unrollIdx);
                  }
                  else{
                    if(j>0){
                      stk->slot_storage[cur_slot][unrollIdx][s] = path(stk, pat, j, stk->uiter[j + 1]);
                    }
                    else{
                      stk->slot_storage[cur_slot][unrollIdx][s] = path(stk, pat, j, stk->uiter[1]);
                    }
                  }
                  stk->slot_size[cur_slot][unrollIdx]++;
                }
              }
            }
            __syncwarp();
            continue;
          }
          compute_set<true>(&arg[wid]);

          if(EDGE_INDUCED && !LABELED){
            for (int j = level-1; j >= -1; j--)
            {
              int unrollIdx = threadIdx.x % WARP_SIZE;
              if(unrollIdx < arg[wid].num_sets)
              {
                if(level==1){
                  arg[wid].set2[unrollIdx] = path_address(stk, pat, j, unrollIdx);
                }
                else{
                  if(j==level-1){
                    arg[wid].set2[unrollIdx] = path_address(stk, pat, j, unrollIdx);
                    arg[wid].set2_size[unrollIdx] = 1;
                  }
                  else{
                    if(j>0){
                      arg[wid].set2[unrollIdx] = path_address(stk, pat, j, stk->uiter[j + 1]);
                    }
                    else{
                      arg[wid].set2[unrollIdx] = path_address(stk, pat, j, stk->uiter[1]);
                    }
                  }
                }
                arg[wid].set1[unrollIdx] = &(stk->slot_storage[cur_slot][unrollIdx][0]);
                arg[wid].res[unrollIdx] = &(stk->slot_storage[cur_slot][unrollIdx][0]);
                arg[wid].set1_size[unrollIdx] = stk->slot_size[cur_slot][unrollIdx];
                arg[wid].res_size[unrollIdx] = &(stk->slot_size[cur_slot][unrollIdx]);
                arg[wid].set2_size[unrollIdx] = 1;
                arg[wid].level = level;
                arg[wid].pat = pat;
              }
              __syncwarp();
              compute_set<true>(&arg[wid]);
            }
          }

          if (!EDGE_INDUCED) {
            for (pattern_node_t j = level - 3; j >= -1; j--) {
              graph_node_t t = path(stk, pat, j, stk->uiter[(j > 0 ? j + 1 : 1)]);

              for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {
                arg[wid].set1[k] = &(stk->slot_storage[i][k][0]);
                arg[wid].set2[k] = &g->colidx[g->rowptr[t]];
                arg[wid].res[k] = &(stk->slot_storage[i][k][0]);
                arg[wid].set1_size[k] = stk->slot_size[i][k];
                arg[wid].set2_size[k] = (set_size_t)(g->rowptr[t + 1] - g->rowptr[t]);
                arg[wid].res_size[k] = &(stk->slot_size[i][k]);
              }
              //arg[wid].cached = true;
              arg[wid].level = level;
              arg[wid].pat = pat;
              compute_set<true>(&arg[wid]);
            }
          }
          for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++) stk->slot_size[cur_slot][k] = 0;
        }
        else {

          //pattern_node_t old_slot_idx = (pat->set_ops[i] & 0x1F);
          pattern_node_t slot_idx = pat->new_set_ops[i] & 0x1F;
          //int stack_num=(pat->new_set_ops[old_slot_idx]&0xf00)>>8;
          // pattern_node_t slot_idx = (pat->new_set_ops[i] & 0x1F)+stk->giter[level-1]%stack_num;
          int stack_num=(pat->new_set_ops[i]&0xf00)>>8;
          pattern_node_t cur_slot=pat->slotptr[i]+stk->giter[level]%stack_num;
      
          if (pat->new_set_ops[i] & 0x40) { //INTE N(V(l-1))作为第二个操作数，做交
            arg[wid].level = level;
            arg[wid].pat = pat;
            int insert_flags=0;
            if(pat->new_set_ops[i]&0x80000){
              int cpy_idx=(pat->new_set_ops[i]&0x7c000)>>14;
              for(int k=0;k<arg[wid].num_sets;k++){
                // int size=stk->slot_size[cpy_idx][k][1];
                int size=stk->slot_size[cpy_idx][k];
                graph_node_t ub_index=upper_bound(&stk->slot_storage[cpy_idx][k][0],size,arg[wid].ub[k]);
                int copy_size=(ub_index<size?ub_index:size);
                CpyMem(&stk->slot_storage[cur_slot][k][0],&stk->slot_storage[cpy_idx][k][0],copy_size);
                stk->slot_size[cur_slot][k]=copy_size;
                //stk->slot_size[cur_slot][k][1]=copy_size;
              } 
              for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++){
                stk->slot_size[cur_slot][k] = 0;
                //stk->slot_size[cur_slot][k][1] = 0;
              }   
              continue; 
            }
            
            for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {
              
              // arg[wid].set1_size[k]=0;
              // arg[wid].set2_size[k]=0;
              //改path
              graph_node_t t0 = path(stk, pat, level - 1, k);
              graph_node_t* neighbor = &g->colidx[g->rowptr[t0]];
              graph_node_t neighbor_size = (graph_node_t)(g->rowptr[t0 + 1] - g->rowptr[t0]);
              graph_node_t t1;
              if (level > 1) {
                if(pat->new_set_ops[i]&0x1000){
                  //t1=stk->vertices[slot_idx][stk->uiter[level-1]];
                  t1=stk->slot_storage[slot_idx][stk->uiter[level-1]][0];
                }else{
                arg[wid].set2[k] = &(stk->slot_storage[slot_idx][stk->uiter[level - 1]][0]);
                arg[wid].set2_size[k] = stk->slot_size[slot_idx][stk->uiter[level - 1]];
                }
              }
              else {
                t1 = path(stk, pat, -1, k);
              }
            if(pat->new_set_ops[i]&0x1000){
              stk->intsec[0]++;
              unsigned hash_index=MyHash(t0,t1)/BUCKET_SIZE;
              int check_result=CheckRepeated(t0,t1,hash_index,global_stk_intsec,hash_table);
              bool writing=0;
              if(check_result!=-1){
                stk->intsec[1]++;
                int bucket_offset=check_result>>3;
                int value_index=check_result&0x7;
                hash_t value=hash_table[hash_index+bucket_offset].value[value_index];
                int global_stk_idx=(value&((1<<(HASH_GL_BIT+HASH_SL_BIT+HASH_K_BIT))-1))>>(HASH_SL_BIT+HASH_K_BIT);
                int rp_slot_idx=(value&((1<<(HASH_SL_BIT+HASH_K_BIT))-1))>>HASH_K_BIT;
                int kk=value&((1<<HASH_K_BIT)-1);
                // unsigned rp_slot_idx=((pat->set_ops[ll]&(1<<10))==0?pat->rowptr[ll+1]-1:pat->rowptr[ll]);
                // unsigned size=global_stk[global_stk_idx].slot_size[rp_slot_idx][kk][1];
                unsigned size=global_stk[global_stk_idx].slot_size[rp_slot_idx][kk];
                graph_node_t ub_index=upper_bound(&global_stk[global_stk_idx].slot_storage[rp_slot_idx][kk][0],size,arg[wid].ub[k]);
                int copy_size=(ub_index<size?ub_index:size);
                int origin_slot=pat->slotptr[i]+stack_num;
                if(copy_size>0){
                  global_stk_intsec[global_wid].intsec_vertices[cur_slot][k][0]=-1;
                  global_stk_intsec[global_wid].intsec_vertices[cur_slot][k][1]=-1;
                  //CpyMem(&stk->slot_storage[cur_slot][k][0],&global_stk[global_stk_idx].slot_storage[rp_slot_idx][kk][0],copy_size);
                  if(acquireReadLock(&global_stk_intsec[global_stk_idx].rwlock[rp_slot_idx][kk])!=-1){
                    //CpyMem(&stk->slot_storage[cur_slot][k][0],&global_stk[global_stk_idx].slot_storage[rp_slot_idx][kk][0],copy_size);
                    int end=CpyRM(&stk->slot_storage[origin_slot][k][0],&global_stk[global_stk_idx].slot_storage[rp_slot_idx][kk][0],copy_size,stk,slot_idx,stk->uiter[level-1]);
                    releaseReadLock(&global_stk_intsec[global_stk_idx].rwlock[rp_slot_idx][kk]);
                    stk->slot_size[origin_slot][k]=end;
                    arg[wid].repeated[k]=true;
                  }else{
                    writing=1;
                  }
                  // else{
                  //   continue;
                  // }
                }
                else{
                
                  stk->slot_size[cur_slot][k]=copy_size;
                  stk->slot_size[origin_slot][k]=0;
                  //stk->slot_size[cur_slot][k][1]=copy_size;
                  arg[wid].repeated[k]=true;
                }
              }if((check_result !=-1 && writing==1)||(check_result==-1)){
                arg[wid].set2[k] = &g->colidx[g->rowptr[t1]];
                arg[wid].set2_size[k] = (set_size_t)(g->rowptr[t1 + 1] - g->rowptr[t1]);
                //if(arg[wid].set2_size[k]>WARP_SIZE*16)arg[wid].key_nodes[k][tid]=g->keycol[g->keyrow[t1]+tid];
              
                arg[wid].set1[k] = neighbor;
                arg[wid].set1_size[k] = neighbor_size;
                insert_flags=insert_flags|(1<<k);

                insert_vertices[wid][k][0]=(t0<t1?t0:t1);
                insert_vertices[wid][k][1]=(t0<t1?t1:t0);
              }
              //arg[wid].res[k] = &(stk->slot_storage[cur_slot][k][0]);
              //arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k][1]);
              //arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k]);
            }else{
              arg[wid].un_intsec=true;
              arg[wid].set1[k] = neighbor;
              arg[wid].set1_size[k] = neighbor_size;
              //arg[wid].res[k] = &(stk->slot_storage[cur_slot][k][0]);
              //arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k]);
            }    
          }
          int getWLock=0;
          for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {
            if(!arg[wid].repeated[k]){
              if(acquireWriteLock(&global_stk_intsec[global_wid].rwlock[cur_slot][k])==0){
                getWLock=getWLock|(1<<k);
                arg[wid].res[k] = &(stk->slot_storage[cur_slot][k][0]);
                arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k]);
              }else{
                int origin_slot=pat->slotptr[i]+stack_num;
                arg[wid].res[k] = &(stk->slot_storage[origin_slot][k][0]);
                arg[wid].res_size[k] = &(stk->slot_size[origin_slot][k]);
                arg[wid].origin[k]=true;
              }
            }
          }
          
            //arg[wid].cached = (level > 1);
            // clock_t start,end;
            // if(tid==0) start=clock();
            compute_set<false>(&arg[wid]);

            for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {
              if(!arg[wid].repeated[k]&&(getWLock&(1<<k)!=0)){
                releaseWriteLock(&global_stk_intsec[global_wid].rwlock[cur_slot][k]);
              }
            }
            // if(tid==0){
            //   end=clock();
            //   //if(level==1&&wid==0)printf("blockid:%d,clock num: %llu\n",blockIdx.x,end-start);
            //   int index;
            //   if(level==1)index=0;
            //   else{
            //     if(i==pat->rowptr[level])index=1;
            //     else index=2;
            //   }
            //   size_t ever=intsec_clock[index];
            // if((end-start)>ever)
            //   intsec_clock[index]=ever+((end-start)-ever)/(compute_set_count[index]+1);
            // else
            //   intsec_clock[index]=ever-(ever-(end-start))/(compute_set_count[index]+1);
            //   compute_set_count[index]++;
            // }
            // if(!(pat->new_set_ops[i]&0x1000)&&tid<arg[wid].num_sets){
            //   stk->slot_size[cur_slot][tid][1]=stk->slot_size[cur_slot][tid][0];
            // }
           if(tid<arg[wid].num_sets && (pat->new_set_ops[i]&0x1000)){
              
              if((insert_flags&(1<<tid))!=0 && (arg[wid].origin[tid]==false)){
                global_stk[global_wid].slot_size[cur_slot][tid]=stk->slot_size[cur_slot][tid];
                // graph_node_t ub_index=upper_bound(&stk->slot_storage[cur_slot][tid][0],stk->slot_size[cur_slot][tid][1],arg[wid].ub[tid]);
                graph_node_t ub_index=upper_bound(&stk->slot_storage[cur_slot][tid][0],stk->slot_size[cur_slot][tid],arg[wid].ub[tid]);
                stk->slot_size[cur_slot][tid]=ub_index;

                // global_stk[global_wid].slot_size[cur_slot][tid][1] = stk->slot_size[cur_slot][tid][1];
                // global_stk[global_wid].slot_size[cur_slot][tid][0] = ub_index;

                global_stk_intsec[global_wid].intsec_vertices[cur_slot][tid][0] = insert_vertices[wid][tid][0];
                global_stk_intsec[global_wid].intsec_vertices[cur_slot][tid][1] = insert_vertices[wid][tid][1];
                
              }
            }
            __syncwarp();
            if((pat->new_set_ops[i]&0x1000)){
              for(int k=0;k<arg[wid].num_sets;k++){
                if((insert_flags&(1<<k))!=0 && !arg[wid].origin[k]){
                  InsertHashTable(insert_vertices[wid][k][0],insert_vertices[wid][k][1],global_wid,cur_slot,k,hash_table);
                }
              }
            }
            if((pat->new_set_ops[i]&0x1000)&&level>=1){    
              int dst_slot=pat->slotptr[i]+stack_num;
              for(int k=0;k<arg[wid].num_sets;k++){
                if((!arg[wid].origin[k])&&(!arg[wid].repeated[k])){
                  if(level>1){
                  int end=CpyRM(&stk->slot_storage[dst_slot][k][0],&stk->slot_storage[cur_slot][k][0],stk->slot_size[cur_slot][k],stk,slot_idx,stk->uiter[level-1]);
                  stk->slot_size[dst_slot][k]=end;
                  }else{
                    CpyMem(&stk->slot_storage[dst_slot][k][0],&stk->slot_storage[cur_slot][k][0],stk->slot_size[cur_slot][k]);
                    stk->slot_size[dst_slot][k]=stk->slot_size[cur_slot][k];
                  }
                }
                //stk->slot_size[dst_slot][k][1]=end;
              }
              for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++) stk->slot_size[dst_slot][k] = 0;
            }
            memset(&arg[wid].repeated[0],false,sizeof(bool)*UNROLL); 
            memset(&arg[wid].origin[0],false,sizeof(bool)*UNROLL); 
            arg[wid].un_intsec=false;

            for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++) stk->slot_size[cur_slot][k] = 0;
          }
          else { //DIFF
            if(pat->new_set_ops[i]&0x80000){
              int cpy_idx=(pat->new_set_ops[i]&0x7c000)>>14;
              for(int k=0;k<arg[wid].num_sets;k++){
                int size=stk->slot_size[cpy_idx][k];
                CpyMem(&stk->slot_storage[cur_slot][k][0],&stk->slot_storage[cpy_idx][k][0],size);
                stk->slot_size[cur_slot][k]=size;
              }
              for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++){
                stk->slot_size[cur_slot][k] = 0;      
              }  
              continue; 
            }
            

            for (graph_node_t k = 0; k < arg[wid].num_sets; k++) {
              graph_node_t* neighbor = NULL;
              graph_node_t neighbor_size = 0;
              if(EDGE_INDUCED && !LABELED){
                neighbor = path_address(stk, pat, level - 1, k);
                neighbor_size = 1;
              }
              if (!EDGE_INDUCED) {
                graph_node_t t = path(stk, pat, level - 1, k);
                neighbor = &g->colidx[g->rowptr[t]];
                neighbor_size = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
              }

              if (level > 1) {
                arg[wid].set1[k] = &(stk->slot_storage[slot_idx][stk->uiter[level - 1]][0]);
                arg[wid].set1_size[k] = stk->slot_size[slot_idx][stk->uiter[level - 1]];
              }
              else {
                graph_node_t t = path(stk, pat, -1, k);
                arg[wid].set1[k] = &g->colidx[g->rowptr[t]];
                arg[wid].set1_size[k] = (set_size_t)(g->rowptr[t + 1] - g->rowptr[t]);
                //if(pat->new_set_ops[i]&0x2000)stk->vertices[cur_slot][k]=t;
                if(pat->new_set_ops[i]&0x2000)stk->slot_storage[cur_slot][k][0]=t;
              }

              arg[wid].set2[k] = neighbor;
              arg[wid].set2_size[k] = neighbor_size;
              arg[wid].res[k] = &(stk->slot_storage[cur_slot][k][0]);
              arg[wid].res_size[k] = &(stk->slot_size[cur_slot][k]);             

            }
            //arg[wid].cached = false;
            arg[wid].level = level;
            arg[wid].pat = pat;
            //if((pat->new_set_ops[i]&0x2000)&&(level==1))continue;
            if((pat->new_set_ops[i]&0x2000)&&(level==1)){
              if(tid<arg[wid].num_sets){
                stk->slot_storage[cur_slot][tid][1]=path(stk, pat, level - 1, tid);
                stk->slot_size[cur_slot][tid]=1; 
              }
              __syncwarp();
              continue;  
            }
            

            compute_set<true>(&arg[wid]);
            for (graph_node_t k = arg[wid].num_sets; k < UNROLL_SIZE(level); k++){
              stk->slot_size[cur_slot][k] = 0;
            }

          }
        }
      }
    }
    stk->iter[level] = 0;
    stk->uiter[level] = 0;
  }

  // __forceinline__ __device__ void respond_across_block(int level, CallStack* stk, Pattern* pat, StealingArgs* _stealing_args) {
  //   if (level > 0 && level <= DETECT_LEVEL) {
  //     if (threadIdx.x % WARP_SIZE == 0) {
  //       int at_level = -1;
  //       int left_task = 0;
  //       for (int l = 0; l < level; l++) {
  //         left_task = stk->slot_size[pat->rowptr[l]][stk->uiter[l]] - stk->iter[l] - stk->uiter[l + 1] - 1;
  //         if (left_task > 0) {
  //           at_level = l;
  //           break;
  //         }
  //       }
  //       if (at_level != -1) {
  //         for (int b = 0; b < GRID_DIM; b++) {
  //           if (b == blockIdx.x) continue;
  //           if (atomicCAS(&(_stealing_args->global_mutex[b]), 0, 1) == 0) {
  //             if (atomicAdd(&_stealing_args->idle_warps[b], 0) == 0xFFFFFFFF) {
  //               __threadfence();

  //               trans_layer(*stk, _stealing_args->global_callstack[b * NWARPS_PER_BLOCK], pat, at_level, INT_MAX);
  //               __threadfence();

  //               atomicSub(_stealing_args->idle_warps_count, NWARPS_PER_BLOCK);
  //               atomicExch(&_stealing_args->idle_warps[b], 0);

  //               atomicExch(&(_stealing_args->global_mutex[b]), 0);
  //               break;
  //             }
  //             atomicExch(&(_stealing_args->global_mutex[b]), 0);
  //           }
  //         }
  //       }
  //     }
  //     __syncwarp();
  //   }
  // }

  __device__ void match(Graph* g, Pattern* pat,
    CallStack* stk, JobQueue* q, size_t* count, Bucket* hash_table, CallStack* global_stk,size_t* intsec,Global_CallStack* global_stk_intsec) {

    pattern_node_t& level = stk->level;
    //if(blockIdx.x * blockDim.x + threadIdx.x==0)printf("level:%d\n",level);
    while (true) {
      // if (threadIdx.x % WARP_SIZE == 0) {
      //   lock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
      // }
      __syncwarp();
      int stk_num=(pat->new_set_ops[pat->rowptr[level]]&0xf00)>>8;
      int slot_idx=pat->slotptr[pat->rowptr[level]]+stk->giter[level]%stk_num;
      if((pat->new_set_ops[pat->rowptr[level]]&0x1000) && (level>=1))       
        slot_idx=pat->slotptr[pat->rowptr[level]]+stk_num;
     
      if (level < pat->nnodes - 2) {
        // if (STEAL_ACROSS_BLOCK) {
        //   respond_across_block(level, stk, pat, _stealing_args);
        // }
        if (stk->uiter[level] == 0 && stk->slot_size[slot_idx][0] == 0) {

          // extend(g, pat, stk, q, level,hash_table,global_stk,intsec_clock,compute_set_count);
          extend(g, pat, stk, q, level,hash_table,global_stk,global_stk_intsec);
          if (level == 0 && stk->slot_size[0][0] == 0) {
            // if (threadIdx.x % WARP_SIZE == 0)
            //   unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
            // __syncwarp();
            break;
          }
        }
        
        if (stk->uiter[level] < UNROLL_SIZE(level)) {
          if (stk->iter[level] < stk->slot_size[slot_idx][stk->uiter[level]]) {
            if (threadIdx.x % WARP_SIZE == 0)
              level++;
            __syncwarp();
          }
          else {
            stk->slot_size[slot_idx][stk->uiter[level]] = 0;
            stk->iter[level] = 0;
            if (threadIdx.x % WARP_SIZE == 0)
              stk->uiter[level]++;
            __syncwarp();
          }
        }
        else {
          stk->uiter[level] = 0;
          if (level > 0) {
            if (threadIdx.x % WARP_SIZE == 0)
              level--;
            if (threadIdx.x % WARP_SIZE == 0){
              stk->iter[level] += UNROLL_SIZE(level + 1);
              stk->giter[level+1]++;
            }
            __syncwarp();
          }
        }
      }
      else if (level == pat->nnodes - 2) {

        extend(g, pat, stk, q, level,hash_table,global_stk,global_stk_intsec);
        for (int j = 0; j < UNROLL_SIZE(level); j++) {
          if (threadIdx.x % WARP_SIZE == 0) {
            *count += stk->slot_size[slot_idx][j];
            //*count += stk->slot_size[pat->rowptr[level]][j];
            if(j==0){
              intsec[0] += stk->intsec[0];
              intsec[1] += stk->intsec[1];
              //intsec[2] += stk->intsec[2];
            }
          }
          __syncwarp();
          //stk->slot_size[pat->rowptr[level]][j] = 0;
          stk->slot_size[slot_idx][j] = 0;
          stk->intsec[0]=0;
          stk->intsec[1]=0;
          //stk->intsec[2]=0;
        }
        stk->uiter[level] = 0;
        if (threadIdx.x % WARP_SIZE == 0)
          level--;
        if (threadIdx.x % WARP_SIZE == 0)
          stk->iter[level] += UNROLL_SIZE(level + 1);
        __syncwarp();
      }
      //__syncwarp();
      // if (threadIdx.x % WARP_SIZE == 0)
      //   unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
      // __syncwarp();
    }
  }



  __global__ void _parallel_match(Graph* dev_graph, Pattern* dev_pattern,
    CallStack* dev_callstack, JobQueue* job_queue, size_t* res,
    Bucket* hash_table,size_t* intsec_count,Global_CallStack* global_stk_intsec) {
    __shared__ Graph graph;
    __shared__ Pattern pat;
    __shared__ CallStack stk[NWARPS_PER_BLOCK];
    __shared__ size_t count[NWARPS_PER_BLOCK];
    //__shared__ bool stealed[NWARPS_PER_BLOCK];
    //__shared__ int mutex_this_block[NWARPS_PER_BLOCK];
    __shared__ size_t intsec[NWARPS_PER_BLOCK][INTSEC_SIZE];
    // __shared__ size_t intsec_clock[NWARPS_PER_BLOCK][3];
    // __shared__ int compute_set_count[NWARPS_PER_BLOCK][3];
    // memset(&intsec_clock[0][0],0,NWARPS_PER_BLOCK*3*sizeof(size_t));
    // memset(&compute_set_count[0][0],0,NWARPS_PER_BLOCK*3*sizeof(size_t));
    

    // __shared__ StealingArgs stealing_args;
    // stealing_args.idle_warps = idle_warps;
    // stealing_args.idle_warps_count = idle_warps_count;
    // stealing_args.global_mutex = global_mutex;
    // stealing_args.local_mutex = mutex_this_block;
    // stealing_args.global_callstack = dev_callstack;

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int global_wid = global_tid / WARP_SIZE;
    int local_wid = threadIdx.x / WARP_SIZE;

    if (threadIdx.x == 0) {
      graph = *dev_graph;
      pat = *dev_pattern;
    }
    __syncthreads();

    if (threadIdx.x % WARP_SIZE == 0) {

      stk[local_wid] = dev_callstack[global_wid];
    }
    __syncwarp();

    auto start = clock64();

    while (true) {
      // match(&graph, &pat, &stk[local_wid], job_queue, &count[local_wid], &stealing_args,hash_table, dev_callstack,&intsec[local_wid][0],&intsec_clock[local_wid][0],&compute_set_count[local_wid][0]);
      match(&graph, &pat, &stk[local_wid], job_queue, &count[local_wid], hash_table, dev_callstack,&intsec[local_wid][0],global_stk_intsec);
      //match(&graph, &pat, &stk[local_wid], job_queue, &count[local_wid], hash_table, dev_callstack,global_stk_intsec);
      __syncwarp();

      //stealed[local_wid] = false;

      // if (STEAL_IN_BLOCK) {

        // if (threadIdx.x % WARP_SIZE == 0) {
        //   stealed[local_wid] = trans_skt(stk, &stk[local_wid], &pat, &stealing_args);
        // }
        // __syncwarp();
      // }

      // if (STEAL_ACROSS_BLOCK) {
      //   if (!stealed[local_wid]) {

      //     __syncthreads();

      //     if (threadIdx.x % WARP_SIZE == 0) {

      //       atomicAdd(stealing_args.idle_warps_count, 1);

      //       lock(&(stealing_args.global_mutex[blockIdx.x]));

      //       atomicOr(&stealing_args.idle_warps[blockIdx.x], (1 << local_wid));

      //       unlock(&(stealing_args.global_mutex[blockIdx.x]));

      //       while ((atomicAdd(stealing_args.idle_warps_count, 0) < NWARPS_TOTAL) && (atomicAdd(&stealing_args.idle_warps[blockIdx.x], 0) & (1 << local_wid)));

      //       if (atomicAdd(stealing_args.idle_warps_count, 0) < NWARPS_TOTAL) {

      //         __threadfence();
      //         if (local_wid == 0) {
      //           stk[local_wid] = (stealing_args.global_callstack[blockIdx.x * NWARPS_PER_BLOCK]);
      //         }
      //         stealed[local_wid] = true;
      //       }
      //       else {
      //         stealed[local_wid] = false;
      //       }
      //     }
      //     __syncthreads();
      //   }
      // }

      // if (!stealed[local_wid]) {
      //   break;
      // }
      break;
    }

    auto stop = clock64();

    if (threadIdx.x % WARP_SIZE == 0) {
      res[global_wid] = count[local_wid];
      // printf("%d\t%ld\t%d\t%d\n", blockIdx.x, stop - start, stealed[local_wid], local_wid);
      //printf("%ld\n", stop - start);
      for(int i=0;i<INTSEC_SIZE;i++)
        intsec_count[global_wid*INTSEC_SIZE+i] = intsec[local_wid][i];
      // for(int i=0;i<3;i++){
      //   clock_count[global_wid*3+i]=intsec_clock[local_wid][i];
      //   gpu_compu_set_count[global_wid*3+i]=compute_set_count[local_wid][i];
      // }
    }

    // if(threadIdx.x % WARP_SIZE == 0)
    //   printf("%d\t%d\t%d\n", blockIdx.x, local_wid, mutex_this_block[local_wid]);
  }
 
}
