/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

void compute(Graph<Empty> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time(); // 计算执行时长

  // dst -> src
  VertexId * parent = graph->alloc_vertex_array<VertexId>();
  // 用于记录访问过的顶点
  VertexSubset * visited = graph->alloc_vertex_subset();
  // 用于记录激活顶点
  VertexSubset * active_in = graph->alloc_vertex_subset();
  VertexSubset * active_out = graph->alloc_vertex_subset();

  visited->clear();
  visited->set_bit(root);
  active_in->clear();
  active_in->set_bit(root);
  graph->fill_vertex_array(parent, graph->vertices);
  parent[root] = root;

  // 活跃顶点的数量
  VertexId active_vertices = 1;

  // begin:直到没有活跃顶点为止
  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_out->clear();
    active_vertices = graph->process_edges<VertexId,VertexId>(
      // sparse mode, push
      [&](VertexId src){
        graph->emit(src, src);  // master发送给mirror
      },
      [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj){
        VertexId activated = 0;
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          // 各个mirror修改自己机器上的数据
          // dst->src 并记录活跃顶点和数量
          if (parent[dst]==graph->vertices && cas(&parent[dst], graph->vertices, src)) {
            active_out->set_bit(dst);
            activated += 1;
          }
        }
        return activated;
      },
      // dense mode, pull
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (visited->get_bit(dst)) return;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_in->get_bit(src)) {
            graph->emit(dst, src); // 各个mirror收集数据发送给master
            break;
          }
        }
      },
      [&](VertexId dst, VertexId msg) {
        if (cas(&parent[dst], graph->vertices, msg)) {
          active_out->set_bit(dst);// master更新数据
          return 1;
        }
        return 0;
      },
      active_in, visited
    );
    // 对所有活跃顶点执行 set visited
    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vtx) {
        visited->set_bit(vtx);
        return 1;
      },
      active_out
    );
    std::swap(active_in, active_out);
  }
  // end:直到没有活跃顶点为止

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(parent, 0);
  if (graph->partition_id==0) {
    VertexId found_vertices = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (parent[v_i] < graph->vertices) {
        found_vertices += 1;
      }
    }
    printf("found_vertices = %u\n", found_vertices);
  }

  graph->dealloc_vertex_array(parent);
  delete active_in;
  delete active_out;
  delete visited;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);
  // 四个入参：
  // <自己>
  // path
  // vertex_num
  // root_vertex_id
  
  if (argc<4) {
    printf("bfs [file] [vertices] [root]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  VertexId root = std::atoi(argv[3]);
  // 加载图
  graph->load_directed(argv[1], std::atoi(argv[2]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
