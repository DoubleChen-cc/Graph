#pragma once

#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include "config.h"

namespace STMatch {

   typedef struct {
    graph_node_t iter[PAT_SIZE];
    graph_node_t uiter[PAT_SIZE];
    graph_node_t giter[PAT_SIZE];
    set_size_t slot_size[MAX_SLOT_NUM][UNROLL];
    graph_node_t (*slot_storage)[UNROLL][GRAPH_DEGREE];
    //graph_node_t vertices[MAX_SLOT_NUM][UNROLL];
    //graph_node_t intsec_vertices[MAX_SLOT_NUM][UNROLL][2];
    graph_node_t intsec[2];
    pattern_node_t level;
  } CallStack;

  struct RWLock {
    int readers;
    int writer;
  };

  typedef struct{
    graph_node_t intsec_vertices[MAX_SLOT_NUM][UNROLL][2];
    RWLock rwlock[MAX_SLOT_NUM][UNROLL];
  }Global_CallStack;

  typedef struct{
    hash_t value[BUCKET_SIZE];
  } Bucket;


/*
  void init() {
    memset(path, 0, sizeof(path));
    memset(iter, 0, sizeof(iter));
    memset(slot_size, 0, sizeof(slot_size));
  }
  */
}