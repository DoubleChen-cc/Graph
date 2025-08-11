#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <cassert>
#include "config.h"


namespace STMatch {

  typedef struct {

    pattern_node_t nnodes = 0;
    int rowptr[PAT_SIZE];
    int degree[PAT_SIZE];
    int slotptr[MAX_SLOT_NUM];
    //bitarray32 slot_labels[MAX_SLOT_NUM];
    bitarray32 partial[MAX_SLOT_NUM];
    set_op_t set_ops[MAX_SLOT_NUM];
    unsigned int new_set_ops[MAX_SLOT_NUM];
    
  } Pattern;


  struct PatternPreprocessor {

    Pattern pat;

    int PatternMultiplicity;
    int adj_matrix_[PAT_SIZE][PAT_SIZE];
    int vertex_order_[PAT_SIZE];
    int order_map_[PAT_SIZE];
    std::vector<std::vector<int>> L_adj_matrix_;
    std::vector<std::vector<int>> board;

    //bitarray32 slot_labels[PAT_SIZE][PAT_SIZE];
    bitarray32 partial[PAT_SIZE][PAT_SIZE];
    set_op_t set_ops[PAT_SIZE][PAT_SIZE];

    std::vector<int> vertex_labels;

    int length[PAT_SIZE];
    int edge[PAT_SIZE][PAT_SIZE];

    PatternPreprocessor(std::string filename) {
      readfile(filename);    
      get_matching_order();
      get_partial_order();
      get_set_ops();
      propagate_partial_order();
      get_labels();
      convert2oned();
      convert_set_ops();

      //std::cout << "Pattern read complete. Pattern size: " << (int)pat.nnodes << std::endl;
    }

    // Pattern* to_gpu() {
    //   Pattern* patcopy;
    //   cudaMalloc(&patcopy, sizeof(Pattern));
    //   cudaMemcpy(patcopy, &pat, sizeof(Pattern), cudaMemcpyHostToDevice);
    //   return patcopy;
    // }

    void readfile(std::string& filename) {
      //std::cout << filename << std::endl;

      std::ifstream fin(filename);
      std::string line;
      while (std::getline(fin, line) && (line[0] == '#'));
      pat.nnodes = 0;

      do {
        std::istringstream sin(line);
        char tmp;
        int v;
        int label;
        sin >> tmp >> v >> label;
        if(LABELED){
          vertex_labels.push_back(label);
        }
        else{
          vertex_labels.push_back(1);
        }
        pat.nnodes++;
      } while (std::getline(fin, line) && (line[0] == 'v'));

      memset(adj_matrix_, 0, sizeof(adj_matrix_));
      do {
        std::istringstream sin(line);
        char tmp;
        int v1, v2;
        int label;
        sin >> tmp >> v1 >> v2 >> label;
        adj_matrix_[v1][v2] = label;
        adj_matrix_[v2][v1] = label;
      } while (getline(fin, line));
    }


    // input from dryadic is alreay reordered 
    void get_matching_order() {
      // int root = 0;
      // int max_degree = 0;
      // for (int i = 0; i < pat.nnodes; i++) {
      //   int d = 0;
      //   for (int j = 0; j < pat.nnodes; j++) {
      //     if (adj_matrix_[i][j] > 0) d++;
      //   }
      //   if (d > max_degree) {
      //     root = i;
      //     max_degree = d;
      //   }
      // }

      // std::vector<int> q;
      // q.push_back(root);
      // int i = 0;
      // std::vector<int> visited(pat.nnodes, 0);
      // while (!q.empty()) {
      //   int a = q.back();
      //   q.pop_back();
      //   if (!visited[a]) {
      //     vertex_order_[i++] = a;
      //   }
      //   visited[a] = 1;
      //   for (int b = 0; b < pat.nnodes; b++) {
      //     if (adj_matrix_[a][b] > 0 && !visited[b])
      //       q.push_back(b);
      //   }
      // }

      // for (int i = 0; i < pat.nnodes; i++)
      //   order_map_[vertex_order_[i]] = i;
    
      for (int i = 0; i < pat.nnodes; i++) {
        vertex_order_[i] = i;
        order_map_[vertex_order_[i]] = i;
      }

      for (int i = 0; i < pat.nnodes; i++) {
        int d = 0;
        for (int j = 0; j < pat.nnodes; j++) {
          if (adj_matrix_[i][j] > 0) d++;
        }
        pat.degree[order_map_[i]] = d;
      }
      /*
       int adj_matrix_tmp[PAT_SIZE][PAT_SIZE];
      memset(adj_matrix_tmp,0,sizeof(adj_matrix_tmp));
      for(int i=0;i<pat.nnodes;i++){
        for(int j=0;j<pat.nnodes;j++){
          adj_matrix_tmp[i][j]=adj_matrix_[vertex_order_[i]][j];
        }
      }
      memset(adj_matrix_,0,sizeof(adj_matrix_));
      for(int i=0;i<pat.nnodes;i++){
        for(int j=0;j<pat.nnodes;j++){
          adj_matrix_[i][j]=adj_matrix_tmp[i][j];
        }
      }
      */
      std::vector<int> vertex_labels_tmp;
      for(int i=0;i<pat.nnodes;i++){
        vertex_labels_tmp.push_back(vertex_labels[vertex_order_[i]]);
      }
      vertex_labels=vertex_labels_tmp;
    }

    void _permutation(
      std::vector<std::vector<int>>& all,
      std::vector<int>& a, int l, int r) {
      // Base case
      if (l == r)
        all.push_back(a);
      else {
        // Permutations made
        for (int i = l; i <= r; i++) {
          // Swapping done
          std::swap(a[l], a[i]);
          // Recursion called
          _permutation(all, a, l + 1, r);
          // backtrack
          std::swap(a[l], a[i]);
        }
      }
    }

    void get_set_ops() {

      board.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      board[0][0] = 1;

      for (int i = 1; i < pat.nnodes - 1; i++) {
        int ops = 0;
        for (int j = 0; j <= i; j++) {
          if (adj_matrix_[vertex_order_[i + 1]][vertex_order_[j]]) ops |= (1 << (i - j));
        }
        board[i][0] = ops;
      }

      memset(length, 0, sizeof(length));
      for (int i = 0; i < pat.nnodes; i++) length[i] = 1;

      memset(set_ops, 0, sizeof(set_ops));
      for (int j = 0; j < pat.nnodes - 1; j++) {
        for (int i = pat.nnodes - 2 - j; i >= 0; i--) {
          // 0 means empty slot in board
          if (board[i][j] == 0) continue;

          int op1 = board[i][j] & 1;
          int op2 = (board[i][j] >> 1);

          if (op2 > 0) {
            bool exist = false;
            // k starts from 1 to make sure candidate sets are not used for computing slots 
            //int startk = ((!LABELED && partial[i - 1][0] == 0) ? 0 : 1);
            int startk = 1;
            for (int k = startk; k < length[i - 1]; k++) {
              if (op2 == board[i - 1][k]) {
                exist = true;
                set_ops[i][j] += k;
                set_ops[i][j] += (op1 << 5);
                break;
              }
            }
            if (!exist) {
              set_ops[i][j] += length[i - 1];
              set_ops[i][j] += (op1 << 5);
              board[i - 1][length[i - 1]++] = op2;
            }
          }
          else {
            set_ops[i][j] |= 0x10;
          }
        }
      }
      // mark the end of slot
      for (int i = 0; i < pat.nnodes - 1; i++) {
        set_ops[i][length[i]] |= 0x80;
      }
    }

    void get_partial_order() {

      std::vector<int> p1;
      for (int i = 0; i < pat.nnodes; i++) {
        p1.push_back(i);
      }
      std::vector<std::vector<int>> permute, valid_permute;
      _permutation(permute, p1, 0, pat.nnodes - 1);
      
      for (auto& pp : permute) {
        std::vector<std::set<int>> adj_tmp(pat.nnodes);
        for (int i = 0; i < pat.nnodes; i++) {
          std::set<int> tp;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 0) continue;
            tp.insert(pp[j]);
          }
          adj_tmp[pp[i]] = tp;
        }
        bool valid = true;
        for (int i = 0; i < pat.nnodes; i++) {
          bool equal = true;
          int c = 0;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 1) {
              c++;
              if (adj_tmp[i].find(j) == adj_tmp[i].end()) equal = false;
            }
          }
          if (!equal || c != adj_tmp[i].size()) {
            valid = false;
            break;
          }
        }
        if (valid)
          valid_permute.push_back(pp);
      }

      PatternMultiplicity = valid_permute.size();

      L_adj_matrix_.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      std::set<std::pair<int, int>> L;
      for (int i = 0; i < pat.nnodes; i++) {
        int v = vertex_order_[i];
        std::vector<std::vector<int>> stabilized_aut;
        for (auto& x : valid_permute) {
          if (x[v] == v) {
            stabilized_aut.push_back(x);
          }
          else {
            L_adj_matrix_[order_map_[v]][order_map_[x[v]]] = 1;
          }
        }
        valid_permute = stabilized_aut;
      }

      memset(partial, 0, sizeof(partial));
      for (int level = 1; level < pat.nnodes; level++) {
        for (int j = level - 1; j >= 0; j--) {
          if (L_adj_matrix_[j][level] == 1) {
            partial[level - 1][0] |= (1 << j);
          }
        }
      }
    }

    int bitidx(bitarray32 a) {
      for (int i = 0; i < 32; i++) {
        if (a & (1 << i)) return i;
      }
      return -1;
    }

    void propagate_partial_order() {
      // propagate partial order of candiate sets to all slots
      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {
          int m = 0;
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            if (set_ops[i + 1][k] & 0x20) {
              // if the slot depends on the current slot and the operation is intersection
              if ((set_ops[i + 1][k] & 0xF) == j) {
                if (partial[i + 1][k] != 0) {
                  // we add the upper bound of that slot to the current slot
                  // the upper bound has to be vertex above level i 
                  m |= (partial[i + 1][k] & (((1 << (i + 1)) - 1)));
                }
                else {
                  m = 0;
                  break;
                }
              }
            }
            else {
              m = 0;
              break;
            }
          }
          partial[i][j] = m;
        }
      }
    }

    void get_labels() {

      //memset(slot_labels, 0, sizeof(slot_labels));

      // for (int i = 0; i < pat.nnodes; i++) {
      //   slot_labels[i][0] = (1 << vertex_labels[i + 1]);
      // }

      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {

          //bitarray32 m = 0;
          //if(j==0) m = pat.partial[i][j];
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            // if the slot depends on the current slot and the operation is intersection
            if ((set_ops[i + 1][k] & 0xF) == j) {
              // we add the upper bound of that slot to the current slot
              // the upper bound has to be vertex above level i 
              //m |= slot_labels[i + 1][k];
            }
          }
          //slot_labels[i][j] = m;
        }
      }
    }

    void convert2oned() {

      int onedidx[PAT_SIZE][PAT_SIZE];
      memset(onedidx, 0, sizeof(onedidx));

      int count = 1;
      pat.rowptr[0] = 0;
      pat.rowptr[1] = 1;
      // this is used for filtering the edges in job queue
      pat.partial[0] = partial[0][0];
      for (int i = 1; i < pat.nnodes - 1; i++) {
        for (int j = 0; j < PAT_SIZE; j++) {
          if (set_ops[i][j] < 0) break;
          onedidx[i][j] = count;
          //pat.slot_labels[count] = slot_labels[i][j];
          pat.partial[count] = partial[i][j];
          int idx = 0;
          if (i > 1) idx = onedidx[i - 1][(set_ops[i][j] & 0x0F)];
          assert(idx < 31);
          pat.set_ops[count] = ((set_ops[i][j] & 0x30) << 1) + idx;
          count++;
        }
        pat.rowptr[i + 1] = count;
      }
      //std::cout << "total number of slots: " << count << std::endl;
      assert(count <= MAX_SLOT_NUM);
      // for(int i=0;i<PAT_SIZE;i++){
      //   std::cout<<pat.rowptr[i]<<' ';
      // }
      // std::cout<<std::endl;
    }
    
    void convert_set_ops(){
      int cpy[MAX_SLOT_NUM],cpy_idx[MAX_SLOT_NUM],v[MAX_SLOT_NUM],intsec[MAX_SLOT_NUM],stack_num[MAX_SLOT_NUM],new_slot_idx[MAX_SLOT_NUM];
      memset(cpy,0,sizeof(int)*MAX_SLOT_NUM);
      memset(cpy_idx,0,sizeof(int)*MAX_SLOT_NUM);
      memset(v,0,sizeof(int)*MAX_SLOT_NUM);
      memset(intsec,0,sizeof(int)*MAX_SLOT_NUM);
      memset(new_slot_idx,0,sizeof(int)*MAX_SLOT_NUM);
      v[0]=1;
      for(int l=1;l<pat.nnodes-1;l++){
        for(int i=pat.rowptr[l];i<pat.rowptr[l+1];i++){
          stack_num[i]=1;
          if(pat.set_ops[i]&0x20){
            if(i!=pat.rowptr[l]&&l<pat.nnodes-2){
              for(int j=pat.rowptr[l+1];j<pat.rowptr[l+2];j++){
                if(((pat.set_ops[j]&0x1F)==i)&&(pat.set_ops[j]&0x40)){
                  v[i]=1;
                }
              }
              for(int j=pat.rowptr[l];j<i;j++){
                if(pat.set_ops[j]==pat.set_ops[i]&&v[i]==v[j]){
                  cpy[i]=1;
                  cpy_idx[i]=j;
                }
              }
            }
          }
          else if(pat.set_ops[i]&0x40){
            for(int j=pat.rowptr[l];j<i;j++){
              if(pat.set_ops[j]==pat.set_ops[i]){
                cpy[i]=1;
                cpy_idx[i]=j;
              }
            }

            int slot_idx=pat.set_ops[i]&0x1F;
            if(v[slot_idx]&&(cpy[i]==0))intsec[i]=1;
          }
          else{
            if(i!=pat.rowptr[l] && l<=1){
              for(int j=pat.rowptr[l+1];j<pat.rowptr[l+2];j++){
                if(((pat.set_ops[j]&0x1F)==i)&&pat.set_ops[j]&0x40){
                  v[i]=1;
                }
              }
            }
            for(int j=pat.rowptr[l];j<i;j++){
              if(pat.set_ops[j]==pat.set_ops[i]){
                cpy[i]=1;
                cpy_idx[i]=j;
              }
            }
          }
        }
      }
       

      int SN=MAX_SN;
      for(int l=pat.nnodes-2;l>=1;l--){
        bool flag=false;
        for(int i=pat.rowptr[l];i<pat.rowptr[l+1];i++){
          if(intsec[i]){
            stack_num[i]=SN;
            flag=true;
          }
        }
        if(flag){
          if(SN>2)SN=SN/2;
          else SN=2;
        }
      }

     
      int count=1;
      for(int l=1;l<pat.nnodes-1;l++){
        for(int i=pat.rowptr[l];i<pat.rowptr[l+1];i++){
          new_slot_idx[i]=count;
          if(pat.set_ops[i]&0x40){
            count+=stack_num[i];
            if(intsec[i]&&l>=1)count++;
          }
          else count++;
        }
      }
      
      ////重新考虑cpy_idx，编码new_set_ops;
      //cpy 1,cpy_idx 5 ,v 1,intsec 1,stack_num 4, set_ops 8
      pat.slotptr[0]=0;
      pat.slotptr[1]=1;
      pat.new_set_ops[0]=(1<<8)|pat.set_ops[0];
      for(int l=1;l<pat.nnodes-1;l++){
        for(int i=pat.rowptr[l];i<pat.rowptr[l+1];i++){
          //printf("cpy:%d,new_slot:%d,v:%d,intsec:%d,stack:%d\n",cpy[i],new_slot_idx[cpy_idx[i]],v[i],intsec[i],stack_num[i]);
          pat.slotptr[i+1]=pat.slotptr[i]+stack_num[i]+((intsec[i]&&l>=1)?1:0);
          int old_slot_idx=pat.set_ops[i]&0x1F;
          int idx=new_slot_idx[old_slot_idx];
          if(intsec[old_slot_idx]&&l>1)idx=idx+stack_num[old_slot_idx];
          int ops=pat.set_ops[i]&0xe0;
          pat.new_set_ops[i]=(cpy[i]<<19)|(new_slot_idx[cpy_idx[i]]<<14)|(v[i]<<13)|(intsec[i]<<12)|(stack_num[i]<<8)|ops|idx;
        }
      }
    }
  };
}