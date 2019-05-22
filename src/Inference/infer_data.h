//
// Created by Mark Van der Merwe, Summer 2018
//

#ifndef CUPGM_PARALLEL_INFERENCE_INFER_DATA_H
#define CUPGM_PARALLEL_INFERENCE_INFER_DATA_H

#include "../header.h"
#include <iostream>
#include <utility>

// Provides helping functions + data structures for our data.

struct device_graph {
  
  int* node_idx_to_incoming_edges;
  int* node_incoming_edges;

  int* node_idx_to_outgoing_edges;
  int* node_outgoing_edges;

  int* edge_idx_to_incoming_edges;
  int* edge_incoming_edges;

  int* edge_idx_to_outgoing_edges;
  int* edge_outgoing_edges;

  int* edge_idx_to_dest_node_idx;

};

struct device_pgm {
  
  int* edge_idx_to_edges_idx;
  double* edges;
  double* workspace;
  
  int* edge_idx_to_edge_factors_idx;
  double* edge_factors;

  int* edge_idx_to_node_factors_idx;
  double* node_factors;
};

// Set up the GPU memory allocations and copy memory in, returning everything wrapped up in a struct for ease of passing + use.
std::pair<device_graph, device_pgm> setup_gpu_data(pgm* pgm);

// Free up the GPU memory used in the device_pgm object for a given problem.
void free_gpu_data(std::pair<device_graph, device_pgm>);

// Same as above but for cpu. This lets us reuse a bunch of code.
std::pair<device_graph, device_pgm> setup_cpu_data(pgm* pgm);

void free_cpu_data(std::pair<device_graph, device_pgm>);

#endif // CUPGM_PARALLEL_INFERENCE_INFER_DATA_H
