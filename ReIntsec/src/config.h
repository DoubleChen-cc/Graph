#pragma once
#include <cstddef>
#include <cstring>
#include <cstdint>
namespace STMatch {

  typedef int graph_node_t;
  typedef long graph_edge_t;
  typedef char pattern_node_t;
  typedef char set_op_t;
  typedef unsigned int bitarray32;
  typedef uint64_t hash_t;
  typedef uint16_t set_size_t;

  inline constexpr size_t PAT_SIZE = 7;
  inline constexpr size_t GRAPH_DEGREE = 4096;
  inline constexpr size_t MAX_SLOT_NUM = 31;
  inline constexpr int INTSEC_SIZE=2;
  inline constexpr int HASH_K_BIT=3; 
  inline constexpr int HASH_SL_BIT=5;
  inline constexpr int HASH_GL_BIT=12;
  //inline constexpr int HASH_XOR_BIT=15;
  inline constexpr int HASH_V_BIT=22;


#include "config_for_ae/table_edge_ulb.h" 

  inline constexpr int GRID_DIM = 82;
  inline constexpr int BLOCK_DIM = 512;
  inline constexpr int WARP_SIZE = 32;
  inline constexpr int NWARPS_PER_BLOCK = (BLOCK_DIM / WARP_SIZE);
  inline constexpr int NWARPS_TOTAL = ((GRID_DIM * BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE);
  inline constexpr unsigned HASH_TABLE_SIZE=0X10000000;
  inline constexpr graph_node_t JOB_CHUNK_SIZE = 8;
  inline constexpr int MAX_SN=8;
  inline constexpr int BUCKET_SIZE=8;
  inline constexpr int CHECK_BUCKET_NUM=2;
  inline constexpr bool KEY_NODE=true;

  //static_assert(2 * JOB_CHUNK_SIZE <= GRAPH_DEGREE); 

  // this is the maximum unroll size

  inline constexpr int DETECT_LEVEL = 1;
  inline constexpr int STOP_LEVEL = 2;

}
