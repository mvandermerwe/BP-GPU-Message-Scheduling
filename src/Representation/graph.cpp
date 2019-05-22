//
// Created by Mark Van der Merwe, Summer 2018
//

#include <iostream>
#include "graph.h"

graph::graph(std::vector<int> node_idx_to_incoming_edges, std::vector<int> node_incoming_edges, std::vector<int> node_idx_to_outgoing_edges, std::vector<int> node_outgoing_edges, std::vector<int> edge_idx_to_incoming_edges, std::vector<int> edge_incoming_edges, std::vector<int> edge_idx_to_outgoing_edges, std::vector<int> edge_outgoing_edges, std::vector<int> edge_idx_to_dest_node_idx) {
  
  this->node_idx_to_incoming_edges = node_idx_to_incoming_edges;
  this->node_incoming_edges = node_incoming_edges;
  this->node_idx_to_outgoing_edges = node_idx_to_outgoing_edges;
  this->node_outgoing_edges = node_outgoing_edges;
  this->edge_idx_to_incoming_edges = edge_idx_to_incoming_edges;
  this->edge_incoming_edges = edge_incoming_edges;
  this->edge_idx_to_outgoing_edges = edge_idx_to_outgoing_edges;
  this->edge_outgoing_edges = edge_outgoing_edges;
  this->edge_idx_to_dest_node_idx = edge_idx_to_dest_node_idx;

}

template <class T>
void print_out_data(std::vector<T> values, std::string label) {
  std::cout << label << std::endl;
  for (T value: values) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}

void graph::print() {
  std::cout << "Graph:" << std::endl;
  print_out_data(node_idx_to_incoming_edges, "Node to incoming edges:");
  print_out_data(node_incoming_edges, "Node incoming edges:");
  print_out_data(node_idx_to_outgoing_edges, "Node to outgoing edges:");
  print_out_data(node_outgoing_edges, "Node outgoing edges:");
  print_out_data(edge_idx_to_incoming_edges, "Edge to incoming edges:");
  print_out_data(edge_incoming_edges, "Edge incoming edges:");
  print_out_data(edge_idx_to_outgoing_edges, "Edge to outgoing edges:");
  print_out_data(edge_outgoing_edges, "Edge outgoing edges:");
  print_out_data(edge_idx_to_dest_node_idx, "Edge to destination node:");
}
