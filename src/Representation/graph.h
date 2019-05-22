//
// Created by Mark Van der Merwe, Summer 2018
//

#ifndef CUPGM_PARALLEL_REPRESENTATION_GRAPH_H_
#define CUPGM_PARALLEL_REPRESENTATION_GRAPH_H_

#include <string>
#include <map>
#include <vector>

// We need our graph for four operations:
// node id -> incoming edges => used in calculating final marginals.
// node id -> outgoing edges => will be used for RS traversals from selected nodes.
// edge id -> incoming edges => used in message update.
// edge id -> outgoing edges => will be used for RBP + RS updates/traversals.
class graph {
  
 public:
  
  std::vector<int> node_idx_to_incoming_edges;
  std::vector<int> node_incoming_edges;

  std::vector<int> node_idx_to_outgoing_edges;
  std::vector<int> node_outgoing_edges;

  std::vector<int> edge_idx_to_incoming_edges;
  std::vector<int> edge_incoming_edges;

  std::vector<int> edge_idx_to_outgoing_edges;
  std::vector<int> edge_outgoing_edges;

  std::vector<int> edge_idx_to_dest_node_idx;

  graph(std::vector<int> node_idx_to_incoming_edges, std::vector<int> node_incoming_edges, std::vector<int> node_idx_to_outgoing_edges, std::vector<int> node_outgoing_edges, std::vector<int> edge_idx_to_incoming_edges, std::vector<int> edge_incoming_edges, std::vector<int> edge_idx_to_outgoing_edges, std::vector<int> edge_outgoing_edges, std::vector<int> edge_idx_to_dest_node_idx);

  void print();

};

#endif // CUPGM_PARALLEL_REPRESENTATION_GRAPH_H_
