//
// Created by Mark Van der Merwe, Fall 2018
//

#include "serial_inference_helpers.h"
#include <vector>
#include <cmath>
#include "../header.h"
#include <omp.h>

double compute_message(pgm* pgm, std::vector<double> &workspace, int edge_id, bool write_to_edges) {
  // Determine where the message value is in the edges array.
  int edges_index = pgm->edge_idx_to_edges_idx[edge_id];
  // Determine the size of the message (i.e., the number of categories).
  int size_of_message = (int) pgm->edges[edges_index];
    
  // First, get the indices of all the relevant messages for this one.
  int relevant_edges_start = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id];
  int relevant_edges_end = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id + 1];
    
  // Update the message.
    
  // Start by clearing the workspace.
  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    workspace[edges_index + 1 + message_idx] = 0;
  }

  // Determine the starting index for the edge factor and for the node factor for this message.
  int edge_factor_idx_start = pgm->edge_idx_to_edge_factors_idx[edge_id];
  int node_factor_idx_start = pgm->edge_idx_to_node_factors_idx[edge_id];

  // Determine if our source node is the first or second node argument of the factor function.
  // Whether we are the first arg determines how to index into the correct factor values.
  int first_arg = pgm->edge_factors[edge_factor_idx_start] == edge_id; // S
  int source_category_count;
  if (first_arg) {
    source_category_count = pgm->edge_factors[edge_factor_idx_start + 1]; // S
  } else {
    source_category_count = pgm->edge_factors[edge_factor_idx_start + 2]; // S
  }
  // Move the factor index start forward to where the actual factor values are.
  edge_factor_idx_start += 3;

  // Loop through each setting of the destination node.
  for (int setting = 0; setting < size_of_message; ++setting) {
    double value = 0.0;

    for (int source_setting = 0; source_setting < source_category_count; ++source_setting) {
      double partial_value = 0.0;

      // Find the edge factor value.
      int edge_factor_idx;
      if (first_arg) {
	edge_factor_idx = edge_factor_idx_start + (source_setting * size_of_message) + setting;
      } else {
	edge_factor_idx = edge_factor_idx_start + source_setting + (setting * source_category_count);
      }
      partial_value = pgm->edge_factors[edge_factor_idx]; // S

      // Find the node factor value.
      partial_value *= pgm->node_factors[node_factor_idx_start + 1 + source_setting]; // S

      // Multiply by incoming messages.
      for (int message_idx = relevant_edges_start; message_idx < relevant_edges_end; ++message_idx) {
	int message_rep_start = pgm->edge_idx_to_edges_idx[pgm->pgm_graph->edge_incoming_edges[message_idx]]; // S
	partial_value *= pgm->edges[message_rep_start + 1 + source_setting]; // S
      }
	
      // Now add the partial value.
      value += partial_value;
    }

    // Write the value to the corresponding location.
    workspace[edges_index + 1 + setting] = value; // S
  }

  // Normalize message by dividing each element by it's sum.
  double sum = 0.0;
  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    double contribution = workspace[edges_index + 1 + message_idx]; // S
    sum += contribution;
  }

  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    double final_value = workspace[edges_index + 1 + message_idx] / sum; // S
    workspace[edges_index + 1 + message_idx] = final_value;
  }

  // Calculate the change in the message.
  double delta = message_delta(pgm->edges, workspace, edges_index);

  // If we want to write straight to edges, switch over now that we've updated the value.
  if (write_to_edges) {
    for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
      pgm->edges[edges_index + 1 + message_idx] = workspace[edges_index + 1 + message_idx];
    }
  }

  return delta;
}

double message_delta(std::vector<double> edges, std::vector<double> workspace, int edge_start) {
  int message_length = workspace[edge_start];

  double sqr_diff_sum = 0.0;
  for (int category = 0; category < message_length; ++category) {
    sqr_diff_sum += std::pow((workspace[edge_start + 1 + category] - edges[edge_start + 1 + category]), 2);
  }
  return std::sqrt(sqr_diff_sum);
}

void compute_marginals(pgm* pgm) {
  for (int node_idx = 0; node_idx < pgm->marginalize_node_ids.size(); ++node_idx) {
    int node_id = pgm->marginalize_node_ids[node_idx];

    int marginal_rep_start = pgm->marginal_to_marginal_rep_idx[node_idx];
    int num_categories = pgm->marginal_rep[marginal_rep_start];

    int node_edges_index = pgm->pgm_graph->node_idx_to_incoming_edges[node_id];
    int num_edges = pgm->pgm_graph->node_idx_to_incoming_edges[node_id + 1] - node_edges_index;

    for (int relevant_edge_idx = 0; relevant_edge_idx < num_edges; ++relevant_edge_idx) {
      for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
	pgm->marginal_rep[marginal_rep_start + 1 + message_idx] *= pgm->edges[pgm->edge_idx_to_edges_idx[pgm->pgm_graph->node_incoming_edges[node_edges_index + relevant_edge_idx]] + 1 + message_idx];
      }
    }

    // Normalize message by dividing each element by it's sum.
    double sum = 0.0;
    for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
      sum += pgm->marginal_rep[marginal_rep_start + 1 + message_idx];
    }
    for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
      pgm->marginal_rep[marginal_rep_start + 1 + message_idx] = pgm->marginal_rep[marginal_rep_start + 1 + message_idx] / sum;
    }
  }
}

void print_doubles(std::vector<double> values) {
  for (double value: values) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}
