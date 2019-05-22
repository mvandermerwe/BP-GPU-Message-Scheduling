//
// Created by Mark Van der Merwe, Summer 2018
//

#include "pgm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <boost/algorithm/string.hpp>

pgm::pgm(std::string input_file) {

  // Parse in the pgm from given input file.
  // Format:
  // ---------------------------
  // Nodes:
  // <name>,{<local factor function>}
  // ...
  //
  // Edges:
  // <node1>,<node2>,{<edge factor function>}
  // ...
  //
  // Marginals:
  // <node1>,...,<noden>
  // ---------------------------
  // Factor functions should be comma separated doubles.
  // Marginals should be comma separated node names representing the nodes you wish to get the marginal for.

  // Helper data structures.
  // TODO: handle faulty input files.
  std::map <std::string, int> node_name_to_id;
  std::map <int, int> node_id_category_size;

  std::map <int, std::pair<int, int>> edge_id_to_nodes;
  std::map <int, std::vector<int>> edge_dest_to_edge_ids;
  std::map <int, std::vector<int>> edge_src_to_edge_ids;

  // Create the graph vectors here and pass them on.
  std::vector<int> node_idx_to_incoming_edges;
  std::vector<int> node_incoming_edges;
  std::vector<int> node_idx_to_outgoing_edges;
  std::vector<int> node_outgoing_edges;
  std::vector<int> edge_idx_to_incoming_edges;
  std::vector<int> edge_incoming_edges;
  std::vector<int> edge_idx_to_outgoing_edges;
  std::vector<int> edge_outgoing_edges;
  std::vector<int> edge_idx_to_dest_node_idx;

  // Open a stream to the file containing the pgm data.
  std::ifstream ifs;
  ifs.open(input_file, std::ifstream::in);

  //-------------------//
  //    Read in PGM    //
  //-------------------//

  // Read the node lines in.
  std::string input;
  std::getline(ifs, input);
  //  if (input != "Nodes:")

  std::getline(ifs, input);
  while (input != "") {
    std::string name = input.substr(0, input.find(","));
    int node_id = node_name_to_id.size();
    node_name_to_id[name] = node_id;

    int start_idx = input.find("{") + 1;
    int end_idx = input.find("}");
    std::string string_factor = input.substr(start_idx, end_idx - start_idx);
    std::vector<std::string> string_factor_split;
    boost::split(string_factor_split, string_factor, boost::is_any_of(","));

    // node_factors
    // Write the node factor.
    // Node factor encoding is same as edge encoding: #cat, {#cat values}
    node_idx_to_node_factor_idx[node_id] = node_factors.size();
    node_factors.push_back(string_factor_split.size()); // Write the number of categories.
    for(std::string value: string_factor_split) {
      node_factors.push_back(std::stod(value));
    }

    // Write the category size.
    node_id_category_size[node_id] = string_factor_split.size();
    std::getline(ifs, input);
  }

  std::getline(ifs, input);
  //if (input != "Edges:");

  int edge_id_counter = 0;
  std::getline(ifs, input);
  while(input != "") {
    // Each edge really represents two edges (i -> j and j -> i).
    std::string names_string = input.substr(0, input.find("{"));
    std::vector<std::string> names;
    boost::split(names, names_string, boost::is_any_of(","));
    int node1_id = node_name_to_id[names[0]];
    int node2_id = node_name_to_id[names[1]];
    int node1_categories = node_id_category_size[node1_id];
    int node2_categories = node_id_category_size[node2_id];

    int forward_edge_id = edge_id_counter;
    int backward_edge_id = edge_id_counter + 1;
    edge_id_counter += 2;
    edge_dest_to_edge_ids[node2_id].push_back(forward_edge_id);
    edge_dest_to_edge_ids[node1_id].push_back(backward_edge_id);
    edge_src_to_edge_ids[node1_id].push_back(forward_edge_id);
    edge_src_to_edge_ids[node2_id].push_back(backward_edge_id);

    // Add to helper nodes_to_edge_id.
    edge_id_to_nodes[forward_edge_id] = std::pair<int,int>(node1_id, node2_id);
    edge_id_to_nodes[backward_edge_id] = std::pair<int,int>(node2_id, node1_id);

    int start_idx = input.find("{") + 1;
    int end_idx = input.find("}");
    std::string string_factor = input.substr(start_idx, end_idx - start_idx);
    std::vector<std::string> string_factor_split;
    boost::split(string_factor_split, string_factor, boost::is_any_of(","));

    // edge_idx_to_edge_factors_idx + edge_factors
    // Note this happens twice because both the forward and backward edge share
    // the edge factor.
    edge_idx_to_edge_factors_idx.push_back(edge_factors.size());
    edge_idx_to_edge_factors_idx.push_back(edge_factors.size());

    // VE: track node to (neighboring node, edge factor):
    node_idx_to_edge_factor_idx[node1_id].push_back(std::make_pair(node2_id, edge_factors.size()));
    node_idx_to_edge_factor_idx[node2_id].push_back(std::make_pair(node1_id, edge_factors.size()));

    // Write the edge factor.
    // fwd_edge_id, i#cat, j#cat, {i#cat * j#cat values}
    edge_factors.push_back(forward_edge_id);
    edge_factors.push_back(node1_categories);
    edge_factors.push_back(node2_categories);
    for(std::string value: string_factor_split) {
      edge_factors.push_back(std::stod(value));
    }

    // edge_idx_edges_idx + edges
    // Add edge encoding.
    // First edge: i->j - |j| categories.
    edge_idx_to_edges_idx.push_back(edges.size());
    edges.push_back(node2_categories);
    for (int i = 0; i < node2_categories; ++i) {
      edges.push_back(1.0/(double) node2_categories);
    }

    // Second edge: j->i - |i| categories.
    edge_idx_to_edges_idx.push_back(edges.size());
    edges.push_back(node1_categories);
    for (int i = 0; i < node1_categories; ++i) {
      edges.push_back(1.0/(double) node1_categories);
    }

    // edge_idx_to_node_factors_idx
    // Add the node_factor indices for these edges.
    edge_idx_to_node_factors_idx.push_back(node_idx_to_node_factor_idx[node1_id]);
    edge_idx_to_node_factors_idx.push_back(node_idx_to_node_factor_idx[node2_id]);

    std::getline(ifs, input);
  }
  edge_idx_to_node_factors_idx.push_back(node_factors.size());
  edge_idx_to_edge_factors_idx.push_back(edge_factors.size());

  // Read in the marginals.
  std::getline(ifs, input);
  //if (input != "Marginals:")

  std::getline(ifs, input);
  // Set up the marginal vectors we need.
  std::vector<std::string> marginal_vars;
  boost::split(marginal_vars, input, boost::is_any_of(","));
  for (std::string marginal_var: marginal_vars) {
    if (marginal_var == "")
      continue;
    int marginal_id = node_name_to_id[marginal_var];
    marginalize_node_ids.push_back(marginal_id);
    marginal_to_marginal_rep_idx.push_back(marginal_rep.size());

    // Set the inital marginal representation to be the node factor function.
    int marginal_factor_start = node_idx_to_node_factor_idx[marginal_id];
    int marginal_size = node_factors[marginal_factor_start];
    marginal_rep.push_back(marginal_size);
    for (int index = marginal_factor_start + 1; index < marginal_factor_start + 1 + marginal_size; ++index) {
      marginal_rep.push_back(node_factors[index]);
    }
  }

  //-------------------//
  // Create the Graph  //
  //-------------------//

  // node_idx_to_incoming_edges + node_incoming_edges
  // node_idx_to_outgoing_edges + node_outgoing_edges
  // Get edge id of all edges corresponding in a given node - we can simply get this from the edge_dest_to_edge_ids/edge_src_to_edge_ids maps.
  for (int node_id = 0; node_id < node_name_to_id.size(); ++node_id) {
    // For each node, append the incoming/outgoing edge ids.
    node_idx_to_incoming_edges.push_back(node_incoming_edges.size());
    node_incoming_edges.insert(node_incoming_edges.end(), edge_dest_to_edge_ids[node_id].begin(), edge_dest_to_edge_ids[node_id].end());
    node_idx_to_outgoing_edges.push_back(node_outgoing_edges.size());
    node_outgoing_edges.insert(node_outgoing_edges.end(), edge_src_to_edge_ids[node_id].begin(), edge_src_to_edge_ids[node_id].end());
  }
  // Append the size to the end to give upper bound for last value.
  node_idx_to_incoming_edges.push_back(node_incoming_edges.size());
  node_idx_to_outgoing_edges.push_back(node_outgoing_edges.size());

  // edge_idx_to_incoming_edges + edge_incoming_edges
  // edge_idx_to_outgoing_edges + edge_outgoing_edges
  // Get edges effecting by using the incoming edges to the source node of the edge and removing self.
  for (int edge_id = 0; edge_id < edge_id_counter; ++edge_id) {
    std::pair<int,int> node_ids = edge_id_to_nodes[edge_id];
    int source_node_id = node_ids.first;
    int dest_node_id = node_ids.second;

    // Push the dest node id for this edge.
    edge_idx_to_dest_node_idx.push_back(dest_node_id);

    // Determine the backward version of this edge. This helps us ensure that we don't add the backwards version of ourselves to effecting edges.
    int backward_edge_id = edge_id % 2 == 0 ? edge_id + 1 : edge_id - 1;

    edge_idx_to_incoming_edges.push_back(edge_incoming_edges.size());
    edge_idx_to_outgoing_edges.push_back(edge_outgoing_edges.size());

    for (int idx = node_idx_to_incoming_edges[source_node_id]; idx < node_idx_to_incoming_edges[source_node_id + 1]; ++idx) {
      int edge = node_incoming_edges[idx];
      if (edge != backward_edge_id) {
	edge_incoming_edges.push_back(edge);
      }
    }

    for (int idx = node_idx_to_outgoing_edges[dest_node_id]; idx < node_idx_to_outgoing_edges[dest_node_id + 1]; ++idx) {
      int edge = node_outgoing_edges[idx];
      if (edge != backward_edge_id) {
	edge_outgoing_edges.push_back(edge);
      }
    }
  }
  // Append the size to the end to give upper bound for last value.
  edge_idx_to_incoming_edges.push_back(edge_incoming_edges.size());
  edge_idx_to_outgoing_edges.push_back(edge_outgoing_edges.size());

  // Create the graph.
  pgm_graph = new graph(node_idx_to_incoming_edges, node_incoming_edges, node_idx_to_outgoing_edges, node_outgoing_edges, edge_idx_to_incoming_edges, edge_incoming_edges, edge_idx_to_outgoing_edges, edge_outgoing_edges, edge_idx_to_dest_node_idx);

  ifs.close();
}

template <class T>
void print_out_data(std::vector<T> values, std::string label) {
  std::cout << label << std::endl;
  for (T value: values) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}

int pgm::num_edges() {
  return edge_idx_to_edges_idx.size();
}

void pgm::print() {
  print_out_data(edge_idx_to_edges_idx, "Edge id to edges:");
  print_out_data(edges, "Edges: ");
  print_out_data(edge_idx_to_edge_factors_idx, "Edge id to edge factor:");
  print_out_data(edge_factors, "Edge Factors:");
  print_out_data(edge_idx_to_node_factors_idx, "Edge id to node factor:");
  print_out_data(node_factors, "Node Factors:");
  print_out_data(marginalize_node_ids, "Marginalize node ids:");
  print_out_data(marginal_to_marginal_rep_idx, "Marginal id to marginal rep idx:");
  print_out_data(marginal_rep, "Marginal rep:");
  pgm_graph->print();
}
