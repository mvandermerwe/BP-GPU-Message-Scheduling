//
// Created by Mark Van der Merwe, Summer 2018
//

#ifndef CUPGM_PARALLEL_REPRESENTATION_PGM_H_
#define CUPGM_PARALLEL_REPRESENTATION_PGM_H_

#include <string>
#include <map>
#include <vector>
#include "graph.h"

class pgm {

 public:

  // Represent the messages on edges.
  std::vector<int> edge_idx_to_edges_idx; // Get the position in the actual edges representation array.
  std::vector<double> edges; // Store the representations for the values.

  // Represent the factors on edges.
  std::vector<int> edge_idx_to_edge_factors_idx; // Get the edge factor for the given edge.
  std::vector<double> edge_factors; // Store the representations for the edge factors.

  // Represent the node factors.
  std::vector<int> edge_idx_to_node_factors_idx; // Get the node factor for the given edge.
  std::vector<double> node_factors; // Store the representations for the node factors.

  // Nodes to marginalize over.
  std::vector<int> marginalize_node_ids; // List of node ids to marginalize.
  std::vector<int> marginal_to_marginal_rep_idx; // Convert from marginal id to it's final belief representation.
  std::vector<double> marginal_rep; // Final belief representations.

  // The following are to be used for Variable Elimination only.
  // Node to the start of its node factor.
  std::map <int, int> node_idx_to_node_factor_idx;
  // Node to list of (neighboring node, edge factor).
  std::map <int, std::vector<std::pair<int, int>>> node_idx_to_edge_factor_idx;

  // Represent the structure of the PGM.
  graph* pgm_graph;

  pgm(std::string input_file);

  int num_edges();

  void print();

};

#endif // CUPGM_PARALLEL_REPRESENTATION_PGM_H_
